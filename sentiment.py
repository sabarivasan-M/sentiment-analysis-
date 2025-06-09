import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')



# LOADING DATASET
df = pd.read_csv('IMDB Dataset.csv',encoding = 'utf-8',on_bad_lines='skip')

#MAPPING SENTIMENT VALUES
df['sentiment'] = df['sentiment'].map({'positive':1,'negative':0})

#TEXT PREPROCESSING
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

df['cleaned_review'] = df['review'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

#SPLITTING DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#TRAINING MODEL
model = MultinomialNB()
model.fit(X_train, y_train)

#EVALUATING MODEL
y_pred = model.predict(X_test)

#PRINTING RESULTS
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

#TESTING WITH NEW REVIEW
new_review = "The movie was great! I loved it."
new_review = preprocess_text(new_review)
print("New Review:", new_review)

# Transform the new review using the same vectorizer
new_review_vectorized = vectorizer.transform([new_review])

# Predict the sentiment
prediction = model.predict(new_review_vectorized)
sentiment = "Positive" if prediction[0] == 1 else "Negative"
print("Predicted Sentiment:", sentiment)