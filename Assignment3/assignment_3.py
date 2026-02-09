import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# Download resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Dataset
data = {
    "text": [
        "Artificial Intelligence is transforming industries!",
        "Machine learning models are powerful tools.",
        "Deep learning improves computer vision systems.",
        "Natural language processing enables chatbots."
    ],
    "label": ["Tech", "Tech", "Tech", "AI"]
}

df = pd.DataFrame(data)


# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df["cleaned_text"] = df["text"].apply(clean_text)


# Stopword removal + Lemmatization
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

df["processed_text"] = df["cleaned_text"].apply(preprocess_text)


# Label encoding
label_encoder = LabelEncoder()
df["encoded_label"] = label_encoder.fit_transform(df["label"])


# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["processed_text"])

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)


# Combine features + label
final_df = pd.concat([tfidf_df, df["encoded_label"]], axis=1)


# Save outputs
df.to_csv("cleaned_dataset.csv", index=False)
tfidf_df.to_csv("tfidf_output.csv", index=False)
final_df.to_csv("final_dataset_with_labels.csv", index=False)

print("Processing complete. Files saved.")
