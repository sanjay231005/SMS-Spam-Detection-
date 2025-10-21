import pandas as pd

# Load dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label','message'])
print(df.head())
print(df['label'].value_counts())
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

ps = PorterStemmer()
corpus = []

for msg in df['message']:
    review = re.sub('[^a-zA-Z]', ' ', msg)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

print(corpus[:5])
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=2500)
X = tfidf.fit_transform(corpus).toarray()
y = pd.get_dummies(df['label'], drop_first=True)  # Spam=1, Ham=0
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
import joblib

joblib.dump(model, 'spam_model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
