import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import re

# 1️⃣ Load Dataset
df = pd.read_csv('spam_detector/ml_models/spam.csv', encoding='latin-1')

# Clean up dataset (remove unnamed extra columns if exist)
# Rename columns because dataset uses v1 for label and v2 for message
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df[['label', 'message']]


# 2️⃣ Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['cleaned'] = df['message'].apply(clean_text)

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['label'], test_size=0.2, random_state=42)

# 4️⃣ Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5️⃣ Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6️⃣ Evaluate
preds = model.predict(X_test_tfidf)
print("Model Accuracy:", accuracy_score(y_test, preds))

# 7️⃣ Save both model and vectorizer
with open('spam_detector/ml_models/spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('spam_detector/ml_models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
