import os
import random
import re
import time  # <-- New import for the delay
from google import genai

# Initialize the Gemini client
client = genai.Client()

def grade_definition(word, user_definition, max_retries=3):
    """Sends the word and definition to Gemini, with a retry loop for API errors."""
    prompt = f"""
    You are an expert psychometrist administering a WAIS-5 style vocabulary test.
    The target word is: "{word}"
    The examinee's definition is: "{user_definition}"
    
    Evaluate the definition and score it 0, 1, or 2 points using these strict rules:
    - 2 Points: Complete understanding. A strong synonym, major use, or definitive classification.
    - 1 Point: Partial understanding. Vague, less precise, or describes only a minor feature instead of the core meaning.
    - 0 Points: Incorrect, totally off-base, or just using the word in a sentence without defining it.
    
    You must reply EXACTLY in this format, with no other text:
    SCORE: [0, 1, or 2]
    REASON: [1 short sentence explaining why it earned this score]
    DEFINITION: [Provide the standard dictionary definition of the word]
    SYNONYMS: [Provide 2 to 3 synonyms for the word, separated by commas]
    EXAMPLE: [Write a creative, Kafkaesque example sentence using the word. Make it sound like it belongs in 'The Trial'.]
    """
    
    # NEW: Retry loop to handle 503 Server Errors gracefully
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            break # If successful, break out of the retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt # Wait 1s, then 2s
                print(f"    [Server busy. Retrying in {delay} seconds...]")
                time.sleep(delay)
            else:
                print(f"\n❌ ERROR: The AI examiner is currently unavailable. ({e})")
                return 0, "API Error.", "N/A", "N/A", "N/A"
    
    # Safely parse the LLM's output line by line
    score = 0
    reason = "Could not parse reason."
    definition = "Definition not provided."
    synonyms = "Synonyms not provided."
    example = "Example not provided."
    
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
        elif line_clean.upper().startswith("EXAMPLE:"):
            example = line_clean.split(":", 1)[1].strip()
            
    return score, reason, definition, synonyms, example

def play_vocab_game(word_list_path, mastered_list_path):
    print("\n" + "="*65)
    print("🧠 WAIS-5 Vocabulary Simulator (Kafka Edition) 🧠")
    print("="*65 + "\n")
    
    try:
        with open(word_list_path, 'r', encoding='utf-8') as f:
            all_words = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: Could not find '{word_list_path}'.")
        return

    mastered_words = set()
    if os.path.exists(mastered_list_path):
        with open(mastered_list_path, 'r', encoding='utf-8') as f:
            mastered_words = set([line.strip() for line in f.readlines() if line.strip()])
            
    words_to_test = [w for w in all_words if w not in mastered_words]
    
    if not words_to_test:
        print("🎉 Congratulations! You have scored a 2 on every word on your list!")
        return
        
    print(f"Total Words: {len(all_words)} | Mastered: {len(mastered_words)} | Remaining: {len(words_to_test)}\n")
    
    random.shuffle(words_to_test)
    
    session_score = 0
    max_possible_score = 0
    
    print("Type your definition. Type 'q' to quit at any time.\n")
    
    for i, word in enumerate(words_to_test):
        print("-" * 65)
        print(f"Word {i+1}/{len(words_to_test)}: What does '{word.upper()}' mean?")
        
        user_def = input("Your answer: ").strip()
        
        if user_def.lower() == 'q':
            break
            
        print("Thinking...")
        score, reason, official_def, synonyms, example = grade_definition(word, user_def)
        
        # Implement the WAIS-5 "Query" rule for 1-point answers
        if score == 1:
            print(f"\n[Examiner]: ⚠️ Partial answer. Tell me more about that.")
            clarification = input("Your elaboration: ").strip()
            
            combined_def = f"{user_def}. Furthermore: {clarification}"
            print("Re-evaluating...")
            score, reason, official_def, synonyms, example = grade_definition(word, combined_def)
            
        if score == 2:
            score_emoji = "✅"
        elif score == 1:
            score_emoji = "🟠"
        else:
            score_emoji = "❌"
            
        print(f"\n{score_emoji} FINAL SCORE: {score}/2")
        print(f"📝 EXAMINER NOTES: {reason}")
        print(f"📖 OFFICIAL DEFINITION: {official_def}")
        print(f"🔗 SYNONYMS: {synonyms}")
        print(f"🏰 KAFKAESQUE EXAMPLE: {example}\n")
        
        if score == 2:
            print(f"⭐ Mastered! '{word}' has been saved and won't be asked again.")
            with open(mastered_list_path, 'a', encoding='utf-8') as f:
                f.write(word + "\n")
            mastered_words.add(word)
            
        session_score += score
        max_possible_score += 2
        
    print("\n" + "="*65)
    print("Session Ended!")
    if max_possible_score > 0:
        percentage = (session_score / max_possible_score) * 100
        print(f"Session Accuracy: {session_score} / {max_possible_score} ({percentage:.1f}%)")
        print(f"Total Words Mastered Overall: {len(mastered_words)} / {len(all_words)}")

if __name__ == "__main__":
    word_file = "new_words_the_lawyer.txt" 
    mastered_file = "mastered_words.txt"
    play_vocab_game(word_file, mastered_file)