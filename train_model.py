import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("/Users/mihuldhakad/Documents/fake news/news.csv")  # Should contain columns: title, text, label

# Combine title and text
df['content'] = df['title'] + " " + df['text']

X = df['content']
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# Evaluate
y_pred = model.predict(tfidf_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(model, open('/Users/mihuldhakad/Documents/fake news /model/fake_news_model.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('/Users/mihuldhakad/Documents/fake news /model/tfidf_vectorizer.pkl', 'wb'))
