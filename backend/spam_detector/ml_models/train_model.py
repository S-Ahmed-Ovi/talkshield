import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import re
import os

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# ---------- LOAD DATA ----------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "spam.csv")

df = pd.read_csv(data_path, encoding='latin-1')
# Adjust for correct column names
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df[['label', 'message']]
df['message'] = df['message'].apply(clean_text)

# ---------- SPLIT DATA ----------
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# ---------- TF-IDF VECTORIZE ----------
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------- TRAIN MODEL ----------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ---------- SAVE MODEL ----------
model_path = os.path.join(base_dir, "spam_model.pkl")
vec_path = os.path.join(base_dir, "vectorizer.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(vec_path, "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully!")
