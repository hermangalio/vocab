import os
import uuid
from datetime import datetime, timezone, timedelta
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

DEPLOY_TIME = datetime.now(timezone(timedelta(hours=2))).strftime('%b %d %H:%M GMT+2')

db.init_app(app)

limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"])


@app.context_processor
def inject_deploy_time():
    return {'deploy_time': DEPLOY_TIME}

with app.app_context():
    db.create_all()
    # Migration: add user_profile_id column if missing (SQLite)
    with db.engine.connect() as conn:
        import sqlalchemy
        cols = [row[1] for row in conn.execute(sqlalchemy.text("PRAGMA table_info(word_list)"))]
        if 'user_profile_id' not in cols:
            conn.execute(sqlalchemy.text("ALTER TABLE word_list ADD COLUMN user_profile_id INTEGER REFERENCES user_profile(id)"))
            conn.commit()
        if 'zipf_threshold' not in cols:
            conn.execute(sqlalchemy.text("ALTER TABLE word_list ADD COLUMN zipf_threshold FLOAT"))
            conn.commit()
        profile_cols = [row[1] for row in conn.execute(sqlalchemy.text("PRAGMA table_info(user_profile)"))]
        if 'session_token' not in profile_cols:
            conn.execute(sqlalchemy.text("ALTER TABLE user_profile ADD COLUMN session_token VARCHAR(36)"))
            conn.commit()
            # Backfill existing profiles with unique tokens
            import uuid as _uuid
            rows = conn.execute(sqlalchemy.text("SELECT id FROM user_profile WHERE session_token IS NULL"))
            for row in rows:
                conn.execute(sqlalchemy.text("UPDATE user_profile SET session_token = :token WHERE id = :id"),
                             {'token': str(_uuid.uuid4()), 'id': row[0]})
            conn.commit()
        word_cols = [row[1] for row in conn.execute(sqlalchemy.text("PRAGMA table_info(word)"))]
        if 'mistakes' not in word_cols:
            conn.execute(sqlalchemy.text("ALTER TABLE word ADD COLUMN mistakes INTEGER DEFAULT 0"))
            conn.commit()


# --------------- Profile Helper ---------------

def get_profile():
    """Get the current user's profile from session, or None.

    Verifies session_token to prevent stale cookies from hijacking
    recycled profile IDs after a database reset.
    """
    profile_id = session.get('profile_id')
    token = session.get('session_token')
    if profile_id and token:
        profile = db.session.get(UserProfile, profile_id)
        if profile and profile.session_token == token:
            return profile
    # Mismatch or missing token — clear stale session
    session.pop('profile_id', None)
    session.pop('session_token', None)
    return None


def require_profile(f):
    """Decorator: auto-creates a profile if none exists."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not get_profile():
            token = str(uuid.uuid4())
            profile = UserProfile(session_token=token)
            db.session.add(profile)
            db.session.flush()
            session['profile_id'] = profile.id
            session['session_token'] = token
            db.session.commit()
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


# --------------- Per-List Calibration ---------------

def pick_calibration_words(word_list, count=15):
    """Pick words from a word list, sorted common → rare, denser among rare words.

    Uses a square-root curve so the first few picks skip quickly through common
    words while the later picks sample more densely among rare words — where the
    interesting differentiation happens.
    """
    words = sorted(word_list.words, key=lambda w: w.zipf_score, reverse=True)
    if len(words) <= count:
        return [{'word': w.word, 'zipf': w.zipf_score} for w in words]
    n = len(words) - 1
    indices = [round(n * (i / (count - 1)) ** 0.5) for i in range(count)]
    return [{'word': words[i].word, 'zipf': words[i].zipf_score} for i in indices]


@app.route('/words/<int:list_id>/calibrate')
@require_access_code
@require_profile
def calibrate(list_id):
    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if not wl or wl.user_profile_id != profile.id:
        flash('Word list not found.', 'error')
        return redirect(url_for('index'))
    words = pick_calibration_words(wl)
    return render_template('calibrate.html', words=words, word_list=wl)


@app.route('/words/<int:list_id>/calibrate/submit', methods=['POST'])
@require_access_code
@require_profile
def calibrate_submit(list_id):
    """Receive yes/no answers for calibration words, compute threshold for this word list."""
    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if not wl or wl.user_profile_id != profile.id:
        return jsonify({'error': 'Not found'}), 404

    data = request.get_json()

    # Custom threshold: skip calibration flow entirely
    custom = data.get('custom_threshold')
    if custom is not None:
        threshold = max(0.5, min(7.0, float(custom)))
    else:
        # Compute from yes/no answers
        answers = data.get('answers', [])
        cal_words = pick_calibration_words(wl)

        threshold = 7.0  # default: quiz only common words
        last_yes_zipf = None
        first_no_zipf = None

        for i, known in enumerate(answers):
            if i < len(cal_words):
                zipf = cal_words[i]['zipf']
                if known:
                    last_yes_zipf = zipf
                elif first_no_zipf is None:
                    first_no_zipf = zipf

        if last_yes_zipf is not None and first_no_zipf is not None:
            threshold = (last_yes_zipf + first_no_zipf) / 2
        elif last_yes_zipf is not None:
            threshold = 0.5

    wl.zipf_threshold = threshold
    db.session.commit()
    return jsonify({'threshold': threshold, 'list_id': list_id})


# --------------- Dashboard ---------------

@app.route('/')
@require_access_code
@require_profile
def index():
    profile = get_profile()
    word_lists = WordList.query.filter_by(user_profile_id=profile.id).order_by(WordList.created_at.desc()).all()
    stats = []
    for wl in word_lists:
        total = len(wl.words)
        threshold = wl.zipf_threshold
        in_range = sum(1 for w in wl.words if threshold is not None and w.zipf_score <= threshold) if threshold else 0
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

        # Create word list record, owned by current user
        profile = get_profile()
        wl = WordList(name=name, status='processing', user_profile_id=profile.id)
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
@require_access_code
@require_profile
def processing(list_id):
    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if not wl or wl.user_profile_id != profile.id:
        flash('Word list not found.', 'error')
        return redirect(url_for('index'))
    if wl.status == 'done':
        return redirect(url_for('calibrate', list_id=list_id))
    if wl.status == 'error':
        flash('An error occurred during extraction.', 'error')
        return redirect(url_for('index'))
    return render_template('processing.html', word_list=wl)


@app.route('/status/<int:list_id>')
@require_access_code
@require_profile
def status(list_id):
    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if not wl or wl.user_profile_id != profile.id:
        return jsonify({'status': 'error'})
    return jsonify({'status': wl.status})


# --------------- Word List View ---------------

@app.route('/words/<int:list_id>')
@require_access_code
@require_profile
def word_list(list_id):
    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if not wl or wl.user_profile_id != profile.id:
        flash('Word list not found.', 'error')
        return redirect(url_for('index'))
    words = Word.query.filter_by(word_list_id=list_id).order_by(Word.zipf_score.asc()).all()
    return render_template('word_list.html', word_list=wl, words=words, threshold=wl.zipf_threshold)


@app.route('/words/<int:list_id>/threshold', methods=['POST'])
@require_access_code
@require_profile
def set_threshold(list_id):
    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if not wl or wl.user_profile_id != profile.id:
        flash('Word list not found.', 'error')
        return redirect(url_for('index'))
    val = request.form.get('threshold', type=float)
    if val is not None:
        wl.zipf_threshold = max(0.5, min(7.0, val))
        db.session.commit()
        flash('Threshold updated.', 'success')
    return redirect(url_for('word_list', list_id=list_id))


@app.route('/words/<int:list_id>/delete', methods=['POST'])
@require_access_code
@require_profile
def delete_word_list(list_id):
    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if wl and wl.user_profile_id == profile.id:
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
    if not wl or wl.user_profile_id != profile.id:
        flash('Word list not found.', 'error')
        return redirect(url_for('index'))
    if wl.zipf_threshold is None:
        return redirect(url_for('calibrate', list_id=list_id))
    total = Word.query.filter_by(word_list_id=list_id).filter(Word.zipf_score <= wl.zipf_threshold).count()
    mastered = Word.query.filter_by(word_list_id=list_id, mastered=True).filter(Word.zipf_score <= wl.zipf_threshold).count()
    return render_template('quiz.html', word_list=wl, total=total, mastered=mastered)


@app.route('/quiz/<int:list_id>/next')
@require_access_code
@require_profile
def quiz_next(list_id):
    """Get the next unmastered word within the list's threshold (random)."""
    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if not wl or wl.user_profile_id != profile.id or wl.zipf_threshold is None:
        return jsonify({'done': True})
    word = (Word.query
            .filter_by(word_list_id=list_id, mastered=False)
            .filter(Word.zipf_score <= wl.zipf_threshold)
            .order_by(db.func.random())
            .first())
    if not word:
        return jsonify({'done': True})
    remaining = (Word.query
                 .filter_by(word_list_id=list_id, mastered=False)
                 .filter(Word.zipf_score <= wl.zipf_threshold)
                 .count())
    return jsonify({
        'done': False,
        'word_id': word.id,
        'word': word.word,
        'remaining': remaining,
    })


@app.route('/quiz/<int:list_id>/answer', methods=['POST'])
@require_access_code
@require_profile
@limiter.limit("20 per minute")
def quiz_answer(list_id):
    """Grade a user's definition."""
    from services.grader import grade_definition

    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if not wl or wl.user_profile_id != profile.id:
        return jsonify({'error': 'Not found'}), 404

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

    if result['score'] == 2:
        word.mastered = True
    else:
        word.mistakes = (word.mistakes or 0) + 1

    db.session.commit()

    return jsonify({
        'score': result['score'],
        'reason': result['reason'],
        'definition': result['definition'],
        'synonyms': result['synonyms'],
        'example': result['example'],
        'needs_query': result['score'] == 1,
        'mastered': word.mastered,
    })


@app.route('/quiz/<int:list_id>/query', methods=['POST'])
@require_access_code
@require_profile
@limiter.limit("20 per minute")
def quiz_query(list_id):
    """Re-grade after elaboration (WAIS-5 query rule)."""
    from services.grader import grade_definition

    profile = get_profile()
    wl = db.session.get(WordList, list_id)
    if not wl or wl.user_profile_id != profile.id:
        return jsonify({'error': 'Not found'}), 404

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
    else:
        word.mistakes = (word.mistakes or 0) + 1

    db.session.commit()

    return jsonify({
        'score': result['score'],
        'reason': result['reason'],
        'definition': result['definition'],
        'synonyms': result['synonyms'],
        'example': result['example'],
        'mastered': word.mastered,
    })


if __name__ == '__main__':
    app.run(debug=True)
