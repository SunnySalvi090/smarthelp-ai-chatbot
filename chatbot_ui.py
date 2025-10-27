# chatbot_ui.py
# Upgraded SmartHelp AI: robust to typos & near-matches
import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# -------------------------
# 1) Load & prepare data
# -------------------------
df = pd.read_csv("customer_support_data.csv")  # ensure clean CSV with question,answer
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# -------------------------
# 2) Utility preprocessing
# -------------------------
spell = SpellChecker(language='en')  # simple spell corrector

def basic_clean(text):
    """Lowercase, strip punctuation, collapse spaces"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove punctuation but keep spaces
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text

def spell_correct_text(text):
    """Correct obvious misspellings using pyspellchecker.
       Keeps words that are numbers or short tokens unchanged in many cases."""
    words = text.split()
    corrected = []
    for w in words:
        # skip very short tokens and digits
        if len(w) <= 2 or w.isdigit():
            corrected.append(w)
            continue
        # If word already in dictionary, keep it
        if w in spell:
            corrected.append(w)
            continue
        # get the most likely correction
        cand = spell.correction(w)
        if cand is None:
            corrected.append(w)
        else:
            corrected.append(cand)
    return " ".join(corrected)

def remove_stopwords(text):
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

def preprocess_text(text):
    text = basic_clean(text)
    text = spell_correct_text(text)     # fixes typos like "charege" -> "charge"
    text = remove_stopwords(text)       # optional but helps matching
    return text

# Create a cleaned column (only do once)
df['cleaned_question'] = df['question'].astype(str).apply(preprocess_text)

# -------------------------
# 3) Vectorizers: word + char n-gram
# -------------------------
# Word-level TF-IDF (semantic, word-aware)
word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2))
X_word = word_vectorizer.fit_transform(df['cleaned_question'])

# Char-level TF-IDF (robust to small misspellings)
char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
X_char = char_vectorizer.fit_transform(df['cleaned_question'])

# -------------------------
# 4) Response logic (combined)
# -------------------------
def combined_similarity_answer(user_input, word_weight=0.6, char_weight=0.4, threshold=0.28):
    """
    Steps:
    1. Preprocess (spell-correct & cleanup)
    2. Compute cosine similarity on word & char vectors
    3. Weighted average of similarities
    4. If below threshold, fallback to fuzzy ratio (rapidfuzz)
    """
    cleaned = preprocess_text(user_input)
    if cleaned.strip() == "":
        return "I'm sorry, I didn't understand that. Could you type a bit more detail?"

    # Transform
    u_word = word_vectorizer.transform([cleaned])
    u_char = char_vectorizer.transform([cleaned])

    # Cosine similarities
    sim_word = cosine_similarity(u_word, X_word).flatten()   # shape (n_questions,)
    sim_char = cosine_similarity(u_char, X_char).flatten()

    # Combined similarity score
    sim_combined = word_weight * sim_word + char_weight * sim_char

    best_idx = sim_combined.argmax()
    best_score = sim_combined.max()

    # If confidence is acceptable, return answer and score
    if best_score >= threshold:
        answer = df.loc[best_idx, 'answer']
        # Also prepare short explainability info
        top3_idx = sim_combined.argsort()[-3:][::-1]
        top3 = [(int(i), float(sim_combined[i]), df.loc[i, 'question']) for i in top3_idx]
        return answer, best_score, top3

    # Fallback: fuzzy string matching (for very short or weird queries)
    # We compute token-based ratio between input and stored questions (using basic_clean)
    cleaned_user = basic_clean(user_input)
    best_fuzzy = -1
    best_fuzzy_idx = None
    for i, q in enumerate(df['question'].astype(str)):
        score_f = fuzz.token_set_ratio(cleaned_user, basic_clean(q))  # 0-100
        if score_f > best_fuzzy:
            best_fuzzy = score_f
            best_fuzzy_idx = i

    # convert fuzzy score to 0..1
    fuzzy_norm = best_fuzzy / 100.0
    # if fuzzy gives decent match, use it
    if fuzzy_norm >= 0.6:
        return df.loc[best_fuzzy_idx, 'answer'], fuzzy_norm, [ (int(best_fuzzy_idx), fuzzy_norm, df.loc[best_fuzzy_idx, 'question']) ]

    # otherwise low confidence
    return "I'm sorry, I couldn't find a good answer. Could you rephrase or give more detail?", best_score, []

# -------------------------
# 5) Streamlit UI
# -------------------------
st.set_page_config(page_title="SmartHelp AI Chatbot", page_icon="💬", layout='centered')
st.title("💬 SmartHelp AI — Customer Support Chatbot (typo-tolerant)")
st.write("Ask anything about orders, returns, refunds, delivery, etc. (typo tolerant!)")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append(("You", user_input))
    answer, score, top_matches = combined_similarity_answer(user_input)
    # show the response
    st.session_state.messages.append(("SmartHelp AI", answer))
    # store debug info in session for optional display
    st.session_state.last_score = float(score)
    st.session_state.last_top = top_matches

# show chat
for sender, msg in st.session_state.messages:
    if sender == "You":
        st.markdown(f"*🧑 {sender}:* {msg}")
    else:
        st.markdown(f"*🤖 {sender}:* {msg}")

# Optional: show confidence and top matches for explainability (collapsible)
if st.session_state.get("last_score", None) is not None:
    with st.expander("Show how the bot decided (confidence & top matches)"):
        st.write(f"Confidence score: *{st.session_state.last_score:.3f}*")
        if st.session_state.last_top:
            st.write("Top matches (index, score, question):")
            for idx, sc, q in st.session_state.last_top:
                st.write(f"- ({idx}) score={sc:.3f} → {q}")