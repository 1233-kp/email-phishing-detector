import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

print("🔄 Starting training...")

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("Phishing_Email.csv")
print("✅ Dataset loaded")

# Rename columns
df = df.rename(columns={
    'Email Text': 'message',
    'Email Type': 'label'
})

df = df[['label', 'message']]
df.dropna(inplace=True)
print("✅ Data cleaned")

# Text preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text)
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

df["cleaned"] = df["message"].apply(clean_text)
print("✅ Text processed")

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]
print("✅ Vectorization done")

# Train model
model = MultinomialNB()
model.fit(X, y)
print("✅ Model trained")

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("🎉 Model saved successfully!")