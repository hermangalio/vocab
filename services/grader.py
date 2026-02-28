import os
import re
import time
from google import genai

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
    return _client


def grade_definition(word, user_definition, max_retries=3):
    """Grade a user's definition using Gemini.

    Returns a dict with: score, reason, definition, synonyms, etymology, reflect.
    """
    prompt = f"""
    You are an expert psychometrist administering a WAIS-5 style vocabulary test.
    The target word is: "{word}"
    The examinee's definition is: "{user_definition}"

    Evaluate the definition and score it 0, 1, or 2 points using these strict rules:
    - 2 Points: Complete understanding. A strong synonym, major use, or definitive classification.
    - 1 Point: Partial understanding. Vague, less precise, or describes only a minor feature instead of the core meaning.
    - 0 Points: Incorrect, totally off-base, or just using the word in a sentence without defining it.
    - CIRCULAR DEFINITIONS score 0: only when the answer just restates the word with no new meaning (e.g. "unluckily" → "not lucky", "happiness" → "being happy"). However, a genuine synonym that shares a root IS valid — e.g. "wearisome" → "tired" is a strong synonym (score 2), NOT circular. The test is whether the answer demonstrates understanding of the meaning, not whether it shares a linguistic root.

    You must reply EXACTLY in this format, with no other text:
    SCORE: [0, 1, or 2]
    REASON: [1 short sentence explaining why it earned this score]
    DEFINITION: [Provide the standard dictionary definition of the word]
    SYNONYMS: [Provide 2 to 3 synonyms for the word, separated by commas]
    ETYMOLOGY: [Briefly explain the word's origin — the language it comes from, its root words, and how its meaning evolved. Keep it to 1-2 sentences.]
    REFLECT: [One sentence max. Start with "Think of a time..." and prompt the reader to recall a specific personal moment where this word applies. Be vivid, not generic.]
    """

    response = None
    for attempt in range(max_retries):
        try:
            response = _get_client().models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={'thinking_config': {'thinking_budget': 0}},
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                time.sleep(delay)
            else:
                return {
                    'score': None,
                    'api_error': True,
                    'reason': "Herman's API ran out — go complain to him.",
                    'definition': 'N/A',
                    'synonyms': 'N/A',
                    'etymology': 'N/A',
                    'reflect': 'N/A',
                }

    # Parse the response
    score = 0
    reason = "Could not parse reason."
    definition = "Definition not provided."
    synonyms = "Synonyms not provided."
    etymology = "Etymology not provided."
    reflect = "Reflection not provided."

    for line in response.text.strip().split('\n'):
        line_clean = line.strip()
        if line_clean.upper().startswith("SCORE:"):
            try:
                score = int(re.sub(r"\D", "", line_clean.split(":", 1)[1]))
            except ValueError:
                score = 0
        elif line_clean.upper().startswith("REASON:"):
            reason = line_clean.split(":", 1)[1].strip()
        elif line_clean.upper().startswith("DEFINITION:"):
            definition = line_clean.split(":", 1)[1].strip()
        elif line_clean.upper().startswith("SYNONYMS:"):
            synonyms = line_clean.split(":", 1)[1].strip()
        elif line_clean.upper().startswith("ETYMOLOGY:"):
            etymology = line_clean.split(":", 1)[1].strip()
        elif line_clean.upper().startswith("REFLECT:"):
            reflect = line_clean.split(":", 1)[1].strip()

    return {
        'score': score,
        'reason': reason,
        'definition': definition,
        'synonyms': synonyms,
        'etymology': etymology,
        'reflect': reflect,
    }
