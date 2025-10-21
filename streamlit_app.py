import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

# Load model & vectorizer
model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf.pkl')

ps = PorterStemmer()

# Preprocessing function
def preprocess(msg):
    review = re.sub('[^a-zA-Z]', ' ', msg)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    return ' '.join(review)

st.title("SMS Spam Detection 🚨")

msg = st.text_area("Enter your SMS message:")

if st.button("Predict"):
    msg_clean = preprocess(msg)
    msg_vector = tfidf.transform([msg_clean]).toarray()
    pred = model.predict(msg_vector)[0]
    st.write("Prediction:", "Spam" if pred==1 else "Ham")
