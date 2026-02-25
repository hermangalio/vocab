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


def extract_words_from_pdf(pdf_path, start_page=None, end_page=None, language='en'):
    """Extract rare vocabulary from a PDF, ranked by word frequency.

    Returns a list of (word, zipf_score) tuples sorted by rarity.
    """
    # Step 1: Read PDF
    pages = extract_pages_from_pdf(pdf_path, start_page, end_page)

    # Fix line-break hyphens
    pages = [re.sub(r'-\s*\n\s*', '', page) for page in pages]

    # Step 2: NLP processing & lemmatization
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000

    cleaned_words = []
    for doc in nlp.pipe(pages, batch_size=5):
        for token in doc:
            if token.is_alpha and token.pos_ != "PROPN":
                lemma = token.lemma_.lower()
                if len(lemma) >= 3 or lemma in ['a', 'i']:
                    cleaned_words.append(lemma)

    # Step 3: Dictionary filter
    raw_unique_words = set(cleaned_words)
    spell = SpellChecker()
    valid_words = spell.known(raw_unique_words)

    # Step 4: Rank by frequency
    word_scores = []
    for word in valid_words:
        score = zipf_frequency(word, language)
        word_scores.append((word, score))

    word_scores.sort(key=lambda x: x[1])
    return word_scores
