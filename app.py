import os
from dotenv import load_dotenv
load_dotenv()

import functools
import threading
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from models import db, UserProfile, WordList, Word, QuizAttempt

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///vocab.db'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['PREFERRED_URL_SCHEME'] = 'https'


@app.before_request
def force_https():
    if request.headers.get('X-Forwarded-Proto') == 'http':
        return redirect(request.url.replace('http://', 'https://'), code=301)

# Access code — set via env var, disabled if not set
ACCESS_CODE = os.environ.get('ACCESS_CODE')

db.init_app(app)

limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"])

with app.app_context():
    db.create_all()


# --------------- Vocabulary Percentile Mapping ---------------
# Approximate mapping from zipf threshold → population percentile.
# Lower threshold = knows rarer words = larger vocabulary = higher percentile.
#
# Derived from:
#   1. Brysbaert et al. (2016) "How Many Words Do We Know?"
#      https://pmc.ncbi.nlm.nih.gov/articles/PMC4965448/
#      Key data for 20-year-old native English speakers:
#        5th percentile:  27,100 lemmas
#        50th percentile: 42,000 lemmas
#        95th percentile: 51,700 lemmas
#
#   2. The wordfreq Zipf scale (van Heuven et al., 2014)
#      https://doi.org/10.3758/s13428-013-0403-5
#
# Method: Using wordfreq's top 80k English words, we counted how many words
# exist at each zipf level. Then we matched Brysbaert's vocabulary sizes to
# zipf boundaries:
#   27,100 lemmas (5th pctile)  → zipf boundary ≈ 3.05
#   42,000 lemmas (50th pctile) → zipf boundary ≈ 2.69
#   51,700 lemmas (95th pctile) → zipf boundary ≈ 2.52
#
# Anchors between the empirical points are interpolated; extremes are
# extrapolated. The entire native-adult range spans only ~0.5 zipf points,
# so results outside the 5th–95th range are rough estimates.

PERCENTILE_ANCHORS = [
    (5.0, 0.5),   # ~1,000 lemmas — far below any native adult norm
    (4.0, 1),     # ~7,000 lemmas — well below adult norms
    (3.5, 2),     # ~15,000 lemmas — below most adults
    (3.05, 5),    # ~27,100 lemmas — Brysbaert 5th percentile (empirical)
    (2.87, 25),   # interpolated between 5th and 50th
    (2.69, 50),   # ~42,000 lemmas — Brysbaert 50th percentile (empirical)
    (2.60, 75),   # interpolated between 50th and 95th
    (2.52, 95),   # ~51,700 lemmas — Brysbaert 95th percentile (empirical)
    (2.3, 98),    # extrapolated
    (2.0, 99),    # ~80,000+ lemmas
    (1.5, 99.5),  # extreme vocabulary
]


def threshold_to_percentile(threshold):
    """Convert a zipf threshold to an approximate population percentile."""
    if threshold >= PERCENTILE_ANCHORS[0][0]:
        return PERCENTILE_ANCHORS[0][1]
    if threshold <= PERCENTILE_ANCHORS[-1][0]:
        return PERCENTILE_ANCHORS[-1][1]
    # Linear interpolation between anchors
    for i in range(len(PERCENTILE_ANCHORS) - 1):
        z1, p1 = PERCENTILE_ANCHORS[i]
        z2, p2 = PERCENTILE_ANCHORS[i + 1]
        if z2 <= threshold <= z1:
            t = (z1 - threshold) / (z1 - z2)
            return p1 + t * (p2 - p1)
    return 50


# --------------- Calibration Words ---------------
# 15 words from common → rare, with verified zipf scores

CALIBRATION_WORDS = [
    ('consider', 4.99),
    ('domestic', 4.69),
    ('narrow', 4.39),
    ('modest', 4.03),
    ('triumph', 3.96),
    ('peculiar', 3.67),
    ('candid', 3.45),
    ('conspicuous', 3.35),
    ('eloquent', 3.21),
    ('ephemeral', 2.99),
    ('pernicious', 2.87),
    ('fastidious', 2.64),
    ('perfunctory', 2.45),
    ('obsequious', 2.14),
    ('pusillanimous', 1.78),
]


# --------------- Profile Helper ---------------

def get_profile():
    """Get the current user's profile from session, or None."""
    profile_id = session.get('profile_id')
    if profile_id:
        return db.session.get(UserProfile, profile_id)
    return None


def require_profile(f):
    """Decorator: redirects to /calibrate if no profile exists."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not get_profile():
            return redirect(url_for('calibrate'))
        return f(*args, **kwargs)
    return decorated


# --------------- Access Code ---------------

def require_access_code(f):
    """Decorator: redirects to /login if ACCESS_CODE is set and user hasn't entered it."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if ACCESS_CODE and not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


@app.route('/login', methods=['GET', 'POST'])
def login():
    if not ACCESS_CODE:
        return redirect(url_for('index'))
    if request.method == 'POST':
        code = request.form.get('code', '').strip()
        if code == ACCESS_CODE:
            session['authenticated'] = True
            return redirect(url_for('index'))
        flash('Wrong access code.', 'error')
    return render_template('login.html')


# --------------- Calibration ---------------

@app.route('/calibrate')
@require_access_code
def calibrate():
    words = [{'word': w, 'zipf': z} for w, z in CALIBRATION_WORDS]
    return render_template('calibrate.html', words=words)


@app.route('/calibrate/submit', methods=['POST'])
@require_access_code
def calibrate_submit():
    """Receive yes/no answers for the 15 calibration words, compute threshold."""
    data = request.get_json()
    answers = data.get('answers', [])  # list of booleans, ordered common → rare

    # Find the threshold: zipf score of the last "yes" word
    threshold = 7.0  # default: quiz only common words (user knows nothing)
    last_yes_zipf = None
    first_no_zipf = None

    for i, known in enumerate(answers):
        if i < len(CALIBRATION_WORDS):
            _, zipf = CALIBRATION_WORDS[i]
            if known:
                last_yes_zipf = zipf
            elif first_no_zipf is None:
                first_no_zipf = zipf

    if last_yes_zipf is not None and first_no_zipf is not None:
        # Threshold = midpoint between last known and first unknown
        threshold = (last_yes_zipf + first_no_zipf) / 2
    elif last_yes_zipf is not None:
        # User knows all 15 words — set threshold very low
        threshold = 0.5
    # else: user knows none — threshold stays at 7.0

    # Create or update profile
    profile = get_profile()
    if profile:
        profile.zipf_threshold = threshold
    else:
        profile = UserProfile(zipf_threshold=threshold)
        db.session.add(profile)
        db.session.flush()
        session['profile_id'] = profile.id

    db.session.commit()
    percentile = threshold_to_percentile(threshold)
    return jsonify({'threshold': threshold, 'percentile': round(percentile, 1)})


# --------------- Dashboard ---------------

@app.route('/')
@require_access_code
@require_profile
def index():
    profile = get_profile()
    word_lists = WordList.query.order_by(WordList.created_at.desc()).all()
    stats = []
    for wl in word_lists:
        total = len(wl.words)
        in_range = sum(1 for w in wl.words if w.zipf_score <= profile.zipf_threshold)
        mastered = sum(1 for w in wl.words if w.mastered)
        stats.append({
            'list': wl,
            'total': total,
            'in_range': in_range,
            'mastered': mastered,
            'pct': round(mastered / in_range * 100) if in_range > 0 else 0,
        })
    return render_template('index.html', stats=stats, profile=profile)


# --------------- PDF Upload & Extraction ---------------

@app.route('/upload', methods=['GET', 'POST'])
@require_access_code
@require_profile
def upload():
    if request.method == 'POST':
        file = request.files.get('pdf')
        if not file or not file.filename.endswith('.pdf'):
            flash('Please upload a PDF file.', 'error')
            return redirect(url_for('upload'))

        start_page = request.form.get('start_page', '').strip()
        end_page = request.form.get('end_page', '').strip()

        start = int(start_page) - 1 if start_page else None
        end = int(end_page) if end_page else None

        # Build a descriptive name
        base_name = os.path.splitext(file.filename)[0]
        if start_page and end_page:
            name = f"{base_name} (pages {start_page}-{end_page})"
        elif start_page:
            name = f"{base_name} (pages {start_page}+)"
        else:
            name = f"{base_name} (full)"

        # Save PDF to a temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        file.save(tmp.name)
        tmp.close()

        # Create word list record
        wl = WordList(name=name, status='processing')
        db.session.add(wl)
        db.session.commit()
        wl_id = wl.id

        # Run extraction in background thread
        thread = threading.Thread(
            target=run_extraction,
            args=(app, tmp.name, wl_id, start, end)
        )
        thread.start()

        return redirect(url_for('processing', list_id=wl_id))

    return render_template('upload.html')


def run_extraction(app, pdf_path, wl_id, start_page, end_page):
    """Run PDF extraction in a background thread."""
    from services.extractor import extract_words_from_pdf

    with app.app_context():
        try:
            word_scores = extract_words_from_pdf(pdf_path, start_page, end_page)

            wl = db.session.get(WordList, wl_id)
            for word, score in word_scores:
                wl.words.append(Word(word=word, zipf_score=score))
            wl.status = 'done'
            db.session.commit()
        except Exception as e:
            wl = db.session.get(WordList, wl_id)
            wl.status = 'error'
            db.session.commit()
            print(f"Extraction error: {e}")
        finally:
            os.unlink(pdf_path)


@app.route('/processing/<int:list_id>')
def processing(list_id):
    wl = db.session.get(WordList, list_id)
    if not wl:
        flash('Word list not found.', 'error')
        return redirect(url_for('index'))
    if wl.status == 'done':
        return redirect(url_for('word_list', list_id=list_id))
    if wl.status == 'error':
        flash('An error occurred during extraction.', 'error')
        return redirect(url_for('index'))
    return render_template('processing.html', word_list=wl)


@app.route('/status/<int:list_id>')
def status(list_id):
    wl = db.session.get(WordList, list_id)
    if not wl:
        return jsonify({'status': 'error'})
    return jsonify({'status': wl.status})


# --------------- Word List View ---------------

@app.route('/words/<int:list_id>')
@require_access_code
@require_profile
def word_list(list_id):
    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if not wl:
        flash('Word list not found.', 'error')
        return redirect(url_for('index'))
    words = Word.query.filter_by(word_list_id=list_id).order_by(Word.zipf_score.asc()).all()
    return render_template('word_list.html', word_list=wl, words=words, threshold=profile.zipf_threshold)


@app.route('/words/<int:list_id>/delete', methods=['POST'])
@require_access_code
@require_profile
def delete_word_list(list_id):
    wl = db.session.get(WordList, list_id)
    if wl:
        db.session.delete(wl)
        db.session.commit()
        flash('Word list deleted.', 'success')
    return redirect(url_for('index'))


# --------------- Quiz ---------------

@app.route('/quiz/<int:list_id>')
@require_access_code
@require_profile
def quiz(list_id):
    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if not wl:
        flash('Word list not found.', 'error')
        return redirect(url_for('index'))
    total = Word.query.filter_by(word_list_id=list_id).filter(Word.zipf_score <= profile.zipf_threshold).count()
    mastered = Word.query.filter_by(word_list_id=list_id, mastered=True).filter(Word.zipf_score <= profile.zipf_threshold).count()
    return render_template('quiz.html', word_list=wl, total=total, mastered=mastered, threshold=profile.zipf_threshold)


@app.route('/quiz/<int:list_id>/next')
@require_profile
def quiz_next(list_id):
    """Get the next unmastered word within the user's threshold (random)."""
    profile = get_profile()
    word = (Word.query
            .filter_by(word_list_id=list_id, mastered=False)
            .filter(Word.zipf_score <= profile.zipf_threshold)
            .order_by(db.func.random())
            .first())
    if not word:
        return jsonify({'done': True})
    remaining = (Word.query
                 .filter_by(word_list_id=list_id, mastered=False)
                 .filter(Word.zipf_score <= profile.zipf_threshold)
                 .count())
    return jsonify({
        'done': False,
        'word_id': word.id,
        'word': word.word,
        'remaining': remaining,
        'threshold': profile.zipf_threshold,
    })


def update_threshold(profile, score):
    """Nudge the user's zipf threshold based on quiz performance."""
    if score == 2:
        profile.zipf_threshold = max(0.5, profile.zipf_threshold - 0.15)
    elif score == 0:
        profile.zipf_threshold = min(7.0, profile.zipf_threshold + 0.15)


@app.route('/quiz/<int:list_id>/answer', methods=['POST'])
@require_access_code
@require_profile
@limiter.limit("20 per minute")
def quiz_answer(list_id):
    """Grade a user's definition."""
    from services.grader import grade_definition

    profile = get_profile()
    data = request.get_json()
    word_id = data.get('word_id')
    user_def = data.get('definition', '').strip()

    word = db.session.get(Word, word_id)
    if not word or word.word_list_id != list_id:
        return jsonify({'error': 'Invalid word'}), 400

    result = grade_definition(word.word, user_def)

    # If API errored, don't save anything — let user retry
    if result.get('api_error'):
        return jsonify({
            'api_error': True,
            'reason': result['reason'],
        })

    # Save attempt
    attempt = QuizAttempt(
        word_id=word.id,
        user_definition=user_def,
        score=result['score'],
        reason=result['reason'],
        official_definition=result['definition'],
        synonyms=result['synonyms'],
        example=result['example'],
    )
    db.session.add(attempt)

    # If score is 2, mark as mastered
    if result['score'] == 2:
        word.mastered = True

    # Update threshold (skip if score is 1 — will be re-graded via query)
    if result['score'] != 1:
        update_threshold(profile, result['score'])

    db.session.commit()

    return jsonify({
        'score': result['score'],
        'reason': result['reason'],
        'definition': result['definition'],
        'synonyms': result['synonyms'],
        'example': result['example'],
        'needs_query': result['score'] == 1,
        'mastered': word.mastered,
        'threshold': profile.zipf_threshold,
    })


@app.route('/quiz/<int:list_id>/query', methods=['POST'])
@require_access_code
@require_profile
@limiter.limit("20 per minute")
def quiz_query(list_id):
    """Re-grade after elaboration (WAIS-5 query rule)."""
    from services.grader import grade_definition

    profile = get_profile()
    data = request.get_json()
    word_id = data.get('word_id')
    original_def = data.get('original_definition', '')
    elaboration = data.get('elaboration', '').strip()

    word = db.session.get(Word, word_id)
    if not word or word.word_list_id != list_id:
        return jsonify({'error': 'Invalid word'}), 400

    combined = f"{original_def}. Furthermore: {elaboration}"
    result = grade_definition(word.word, combined)

    # If API errored, don't save anything — let user retry
    if result.get('api_error'):
        return jsonify({
            'api_error': True,
            'reason': result['reason'],
        })

    # Save attempt
    attempt = QuizAttempt(
        word_id=word.id,
        user_definition=combined,
        score=result['score'],
        reason=result['reason'],
        official_definition=result['definition'],
        synonyms=result['synonyms'],
        example=result['example'],
    )
    db.session.add(attempt)

    if result['score'] == 2:
        word.mastered = True

    # Update threshold after final grading
    update_threshold(profile, result['score'])

    db.session.commit()

    return jsonify({
        'score': result['score'],
        'reason': result['reason'],
        'definition': result['definition'],
        'synonyms': result['synonyms'],
        'example': result['example'],
        'mastered': word.mastered,
        'threshold': profile.zipf_threshold,
    })


if __name__ == '__main__':
    app.run(debug=True)
