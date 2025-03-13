import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset (Replace with actual CSV file if needed)
data = {
    'Review': [
        'This product is amazing! Highly recommended.',
        'Worst purchase ever. Do not buy this.',
        'Absolutely love it. Works like a charm!',
        'Not what I expected. Very disappointed.',
        'Decent product for the price.',
        'Terrible quality. Waste of money.',
        'Exceeded my expectations! Will buy again.',
        'Not great, but okay for casual use.',
    ],
    'Sentiment': ['positive', 'negative', 'positive', 'negative', 'neutral', 'negative', 'positive', 'neutral']
}
df = pd.DataFrame(data)

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Apply preprocessing
df['Cleaned_Review'] = df['Review'].apply(preprocess_text)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Cleaned_Review'])

# Convert sentiment labels to numerical values
df['Sentiment_Label'] = df['Sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 2})
y = df['Sentiment_Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Function to predict sentiment of new reviews
def predict_sentiment(review):
    review = preprocess_text(review)
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]
    sentiment_map = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
    return sentiment_map[prediction]

# Example Usage
new_review = "The product is very good and works perfectly!"
print(f'Review: "{new_review}" --> Sentiment: {predict_sentiment(new_review)}')

