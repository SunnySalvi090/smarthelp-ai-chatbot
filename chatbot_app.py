# --- Step 1: Load dataset ---

import pandas as pd   # pandas helps us read CSV files

# read the CSV file
df = pd.read_csv("customer_support_data.csv")

# show first few rows
print(df.head())

# --- Step 2: Preprocess the text ---

import nltk
from nltk.corpus import stopwords
import string

# download stopwords once (this runs only the first time)
nltk.download('stopwords')

def preprocess(text):
    text = text.lower()                                      # lowercase
    text = ''.join([c for c in text if c not in string.punctuation])  # remove punctuation
    tokens = text.split()                                    # split into words
    stop_words = set(stopwords.words('english'))             # common words to ignore
    tokens = [w for w in tokens if w not in stop_words]      # remove stopwords
    return ' '.join(tokens)

# create a cleaned column
df['cleaned_question'] = df['question'].apply(preprocess)
print(df[['question', 'cleaned_question']].head())

# --- Step 3: Vectorization (TF-IDF) ---

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_question'])

print("TF-IDF matrix shape:", X.shape)


# --- Step 4: Chatbot response logic ---

from sklearn.metrics.pairwise import cosine_similarity

def chatbot_response(user_input):
    # clean and vectorize user input
    user_input = preprocess(user_input)
    user_vec = vectorizer.transform([user_input])
    
    # find similarity with all known questions
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()          # index of best match
    score = similarity.max()             # how close it was

    # handle low confidence
    if score < 0.3:
        return "I'm sorry, I didn't understand that. Could you rephrase?"
    else:
        return df['answer'][index]
    
    # --- Step 5: Start chatting ---

print("SmartHelp AI: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("SmartHelp AI: Thank you for chatting with us. Have a nice day!")
        break
    print("SmartHelp AI:", chatbot_response(user_input))