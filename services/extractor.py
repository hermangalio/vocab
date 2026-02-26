import re
import PyPDF2
import spacy
from wordfreq import zipf_frequency
from spellchecker import SpellChecker


def extract_pages_from_pdf(pdf_path, start_page=None, end_page=None):
    """Reads a PDF and extracts text as a list of page strings."""
    pages = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)

        start = start_page if start_page is not None else 0
        end = end_page if end_page is not None else total_pages

        start = max(0, start)
        end = min(total_pages, end)

        for page_num in range(start, end):
            page_text = reader.pages[page_num].extract_text()
            if page_text:
                pages.append(page_text)

    return pages


def extract_words_from_pdf(pdf_path, start_page=None, end_page=None, language='en',
                           on_progress=None):
    """Extract rare vocabulary from a PDF, ranked by word frequency.

    Returns a list of (word, zipf_score) tuples sorted by rarity.
    on_progress(pct) is called with 0-100 to report progress.
    """
    def report(pct):
        if on_progress:
            on_progress(pct)

    # Step 1: Read PDF (0-10%)
    report(5)
    pages = extract_pages_from_pdf(pdf_path, start_page, end_page)
    pages = [re.sub(r'-\s*\n\s*', '', page) for page in pages]
    report(10)

    # Step 2: NLP processing & lemmatization (10-75%)
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000

    cleaned_words = []
    total_pages = len(pages)
    for i, doc in enumerate(nlp.pipe(pages, batch_size=5)):
        for token in doc:
            if token.is_alpha and token.pos_ != "PROPN":
                lemma = token.lemma_.lower()
                if len(lemma) >= 3 or lemma in ['a', 'i']:
                    cleaned_words.append(lemma)
        report(10 + round(65 * (i + 1) / max(total_pages, 1)))

    # Step 3: Dictionary filter (75-90%)
    report(80)
    raw_unique_words = set(cleaned_words)
    spell = SpellChecker()
    valid_words = spell.known(raw_unique_words)
    report(90)

    # Step 4: Rank by frequency (90-100%)
    word_scores = []
    for word in valid_words:
        score = zipf_frequency(word, language)
        word_scores.append((word, score))

    word_scores.sort(key=lambda x: x[1])
    report(100)
    return word_scores
