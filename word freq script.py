import PyPDF2
import re
import spacy
from wordfreq import zipf_frequency
from tqdm import tqdm
from spellchecker import SpellChecker

def extract_pages_from_pdf(pdf_path, start_page=None, end_page=None):
    """Reads a PDF file and extracts text as a list of pages within a specified range."""
    pages = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)
        
        # Set defaults if no pages are provided
        start = start_page if start_page is not None else 0
        end = end_page if end_page is not None else total_pages
        
        # Ensure we don't go out of bounds
        start = max(0, start)
        end = min(total_pages, end)
        
        # Slicing the pages to only extract the ones the user requested
        page_range = list(range(start, end))
        
        for page_num in tqdm(page_range, desc="Extracting pages", unit="page"):
            page_text = reader.pages[page_num].extract_text()
            if page_text:
                pages.append(page_text)
                
    return pages

def rank_words_by_global_frequency(pdf_path, start_page=None, end_page=None, language='en'):
    print(f"\n--- Step 1: Read PDF ---")
    pages = extract_pages_from_pdf(pdf_path, start_page, end_page)
    
    print("Stitching line-break hyphens back together...")
    pages = [re.sub(r'-\s*\n\s*', '', page) for page in pages]
    
    print("\n--- Step 2: NLP Processing & Lemmatization ---")
    print("Loading spaCy language model (this takes a moment)...")
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000 
    
    cleaned_words = []
    
    for doc in tqdm(nlp.pipe(pages, batch_size=5), total=len(pages), desc="Analyzing grammar", unit="page"):
        for token in doc:
            if token.is_alpha and token.pos_ != "PROPN":
                lemma = token.lemma_.lower()
                if len(lemma) >= 3 or lemma in ['a', 'i']:
                    cleaned_words.append(lemma)
    
    print("\n--- Step 3: Dictionary Filter ---")
    raw_unique_words = set(cleaned_words)
    print(f"Found {len(raw_unique_words)} unique base words (lemmas).")
    
    print("Checking words against official English dictionary...")
    spell = SpellChecker()
    valid_words = spell.known(raw_unique_words)
    print(f"Filtered down to {len(valid_words)} valid dictionary words.")
    
    print("\n--- Step 4: Rank Words ---")
    word_scores = []
    for word in tqdm(valid_words, desc="Scoring words", unit="word"):
        score = zipf_frequency(word, language)
        word_scores.append((word, score))
        
    word_scores.sort(key=lambda x: x[1])
    
    return word_scores

if __name__ == "__main__":
    pdf_filename = "the trial version i have as a book.pdf"  # <-- Make sure your PDF name is correct here
    
    print("=== PDF Vocabulary Extractor ===")
    print("Would you like to process the whole book or a specific page range?")
    start_input = input("Enter start page number (or press Enter for the whole book): ")
    
    # Logic to handle user input
    if start_input.strip():
        start_page = int(start_input) - 1 # Subtract 1 because Python lists start at 0
        end_input = input("Enter end page number: ")
        end_page = int(end_input) if end_input.strip() else None
        
        # Dynamically name the files based on the pages selected
        suffix = f"_pages_{start_input}_to_{end_input}"
    else:
        start_page = None
        end_page = None
        suffix = "_full_book"
        
    output_with_scores = f"obscure_words_lemmatized{suffix}.txt"
    output_words_only = f"obscure_words_only{suffix}.txt"
    
    try:
        ranked_words = rank_words_by_global_frequency(pdf_filename, start_page, end_page)
        
        print(f"\nSaving {len(ranked_words)} words to '{output_with_scores}'...")
        with open(output_with_scores, 'w', encoding='utf-8') as f:
            f.write(f"Vocabulary - Ranked by Rarity (Lemmatized & Cleaned)\n")
            f.write("=" * 65 + "\n")
            for word, score in ranked_words:
                f.write(f"{word:<25} Zipf Score: {score}\n")
                
        print(f"Saving words-only list to '{output_words_only}'...")
        with open(output_words_only, 'w', encoding='utf-8') as f:
            for word, _ in ranked_words:
                f.write(f"{word}\n")
                
        print("\nDone! Both text files have been created in your current folder.")
            
    except FileNotFoundError:
        print(f"Error: Could not find the file '{pdf_filename}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")