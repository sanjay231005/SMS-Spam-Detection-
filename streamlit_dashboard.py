import streamlit as st
import joblib
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import nltk
import pandas as pd
from io import BytesIO
import base64

nltk.download('stopwords')

# --- Load model and vectorizer ---
model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf.pkl')
ps = PorterStemmer()

# --- Preprocessing function ---
def preprocess(msg):
    review = re.sub('[^a-zA-Z]', ' ', msg)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    return ' '.join(review)

# --- Streamlit Layout ---
st.title("📊 SMS Spam Detection Dashboard")

# --- Sidebar Test Data Upload ---
st.sidebar.header("Optional Test Data for Metrics")
uploaded_test_file = st.sidebar.file_uploader(
    "Upload CSV with 'message' and 'label' columns for metrics (optional)",
    type="csv"
)

test_data_available = False
single_prediction_data = None  # To store single message result for tabs

if uploaded_test_file is not None:
    test_df = pd.read_csv(uploaded_test_file)
    if 'message' in test_df.columns and 'label' in test_df.columns:
        X_test = test_df['message'].tolist()
        y_test = test_df['label'].tolist()
        test_data_available = True
        st.sidebar.success("Test data loaded successfully!")
    else:
        st.sidebar.warning("CSV must contain 'message' and 'label' columns.")

# --- Tabs ---
tabs = st.tabs(["💬 Live & Batch Prediction", "📈 Metrics & Curves", "📊 Visualizations", "🛠️ Model Architecture"])

# --- TAB 1: Live & Batch Prediction ---
with tabs[0]:
    st.subheader("💬 SMS Prediction")

    # --- Single Message Prediction ---
    st.markdown("### 📝 Predict a Single Message")
    msg = st.text_area("Type your SMS here:")

    if st.button("Predict Message"):
        if msg.strip() == "":
            st.warning("Please type a message!")
        else:
            # Preprocess and vectorize
            msg_clean = preprocess(msg)
            msg_vector = tfidf.transform([msg_clean]).toarray()

            # Predict
            pred = model.predict(msg_vector)[0]
            pred_prob = model.predict_proba(msg_vector)[0]
            pred_label = 'SPAM' if pred else 'HAM'

            # Store for tabs
            single_prediction_data = {
                "message": msg,
                "prediction": pred_label,
                "prob_ham": pred_prob[0],
                "prob_spam": pred_prob[1],
                "processed": msg_clean
            }

            # Display numeric probabilities
            st.markdown(f"**Prediction:** {pred_label}")
            st.markdown(f"**Probability:** Ham: {pred_prob[0]:.2f}, Spam: {pred_prob[1]:.2f}")

            # Probability bar chart
            fig, ax = plt.subplots()
            ax.bar(['Ham', 'Spam'], pred_prob, color=['green', 'red'])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Probabilities")
            st.pyplot(fig)

            # Mini metrics table
            result_df = pd.DataFrame({
                "Message": [msg],
                "Prediction": [pred_label],
                "Prob_Ham": [pred_prob[0]],
                "Prob_Spam": [pred_prob[1]]
            })
            st.table(result_df)

            # Mini confusion-style chart
            st.markdown("**Prediction Visualization (Mini Confusion Style)**")
            cm_fig, cm_ax = plt.subplots()
            cm_ax.bar(['Ham','Spam'], [pred_prob[0], pred_prob[1]], color=['green','red'])
            cm_ax.set_ylim(0,1)
            cm_ax.set_ylabel("Probability")
            cm_ax.set_title("Prediction Probability Distribution")
            st.pyplot(cm_fig)

            # WordCloud for single message
            st.subheader("☁️ WordCloud for This Message")
            wc_text = msg_clean  # already preprocessed
            wc = WordCloud(width=800, height=400, background_color='white').generate(wc_text)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # Download CSV
            def get_download_link(df, filename="single_prediction.csv"):
                towrite = BytesIO()
                df.to_csv(towrite, index=False)
                towrite.seek(0)
                b64 = base64.b64encode(towrite.read()).decode()
                return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Prediction CSV</a>'
            st.markdown(get_download_link(result_df), unsafe_allow_html=True)

    # --- Batch Prediction ---
    st.markdown("### 📂 Batch Prediction via CSV")
    st.markdown("Upload a CSV file with a column named `message` containing SMS texts.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'message' not in df.columns:
            st.error("CSV must contain a 'message' column.")
        else:
            df['processed'] = df['message'].apply(preprocess)
            X_batch = tfidf.transform(df['processed']).toarray()

            # Predictions
            df['prediction'] = model.predict(X_batch)
            df['prob_ham'] = model.predict_proba(X_batch)[:, 0]
            df['prob_spam'] = model.predict_proba(X_batch)[:, 1]
            df['prediction_label'] = df['prediction'].map({0: 'HAM', 1: 'SPAM', False: 'HAM', True: 'SPAM'})

            st.success(f"Batch Prediction Completed: {len(df)} messages processed.")
            st.dataframe(df[['message', 'prediction_label', 'prob_ham', 'prob_spam']])

            # Probability bar chart for batch averages
            avg_probs = [df['prob_ham'].mean(), df['prob_spam'].mean()]
            fig, ax = plt.subplots()
            ax.bar(['Ham', 'Spam'], avg_probs, color=['green','red'])
            ax.set_ylim(0,1)
            ax.set_ylabel("Average Probability")
            ax.set_title("Average Prediction Probabilities (Batch)")
            st.pyplot(fig)

            # Download link
            def get_download_link(df, filename="batch_predictions.csv"):
                towrite = BytesIO()
                df.to_csv(towrite, index=False)
                towrite.seek(0)
                b64 = base64.b64encode(towrite.read()).decode()
                return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Prediction CSV</a>'
            st.markdown(get_download_link(df, "batch_predictions.csv"), unsafe_allow_html=True)

# --- TAB 2: Metrics & Curves ---
with tabs[1]:
    st.subheader("📈 Model Performance Metrics")

    if test_data_available:
        # Use test dataset
        X_test_vector = tfidf.transform([preprocess(x) for x in X_test])
        y_pred = model.predict(X_test_vector)
        y_prob = model.predict_proba(X_test_vector)[:,1]
        y_pred_labels = np.array(['spam' if p else 'ham' for p in y_pred])

        # Classification Report
        report = classification_report(y_test, y_pred_labels, output_dict=True)
        st.markdown("**Classification Report**")
        st.table({
            "Metric": ["Precision", "Recall", "F1-Score", "Support"],
            "Ham": [report['ham']['precision'], report['ham']['recall'], report['ham']['f1-score'], report['ham']['support']],
            "Spam": [report['spam']['precision'], report['spam']['recall'], report['spam']['f1-score'], report['spam']['support']],
            "Average": [report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score'], report['weighted avg']['support']]
        })

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_labels, labels=['ham','spam'])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)

        # ROC & PR curves
        label_mapping = {'ham':0, 'spam':1}
        y_test_numeric = np.array([label_mapping[label] for label in y_test])
        y_prob_numeric = y_prob

        fpr, tpr, _ = roc_curve(y_test_numeric, y_prob_numeric)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0,1], [0,1], color='navy', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        st.pyplot(fig)

        precision, recall, _ = precision_recall_curve(y_test_numeric, y_prob_numeric)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='purple')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        st.pyplot(fig)

    elif single_prediction_data:
        # Use single message for mini metrics
        st.markdown("**Single Message Metrics**")
        df = pd.DataFrame({
            "Message": [single_prediction_data['message']],
            "Prediction": [single_prediction_data['prediction']],
            "Prob_Ham": [single_prediction_data['prob_ham']],
            "Prob_Spam": [single_prediction_data['prob_spam']]
        })
        st.table(df)

        # Mini confusion-style chart
        fig, ax = plt.subplots()
        ax.bar(['Ham','Spam'], [single_prediction_data['prob_ham'], single_prediction_data['prob_spam']], color=['green','red'])
        ax.set_ylim(0,1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probability Distribution")
        st.pyplot(fig)

    else:
        st.info("No data available for metrics.")

# --- TAB 3: Visualizations ---
with tabs[2]:
    st.subheader("📊 Data Visualizations")
    if test_data_available:
        unique, counts = np.unique(y_test, return_counts=True)
        fig, ax = plt.subplots()
        ax.pie(counts, labels=['Ham','Spam'], autopct='%1.1f%%', colors=['green','red'])
        st.pyplot(fig)

        st.subheader("☁️ WordCloud of Spam Messages")
        spam_text = " ".join([preprocess(X_test[i]) for i in range(len(y_test)) if y_test[i]=='spam'])
        wc = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    elif single_prediction_data:
        st.markdown("**Single Message Visualization**")
        # Pie chart for predicted class
        probs = [single_prediction_data['prob_ham'], single_prediction_data['prob_spam']]
        fig, ax = plt.subplots()
        ax.pie(probs, labels=['Ham','Spam'], autopct='%1.1f%%', colors=['green','red'])
        ax.set_title("Prediction Probability Pie Chart")
        st.pyplot(fig)

        # WordCloud
        wc_text = single_prediction_data['processed']
        wc = WordCloud(width=800, height=400, background_color='white').generate(wc_text)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    else:
        st.info("No data available for visualizations.")

# --- TAB 4: Model Architecture ---
with tabs[3]:
    st.subheader("🛠️ SMS Spam Detection Pipeline")
    architecture = """
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, color=lightblue];

        Raw_SMS [label="Raw SMS Input"];
        Preprocessing [label="Preprocessing\\n(remove punctuation, lowercase,\\nstopwords removal, stemming)"];
        Vectorization [label="TF-IDF Vectorization"];
        Model [label="Trained Model\\n(RandomForest / SVM / Logistic Regression)"];
        Prediction [label="Prediction Output\\n(Ham or Spam)"];

        Raw_SMS -> Preprocessing -> Vectorization -> Model -> Prediction;
    }
    """
    st.graphviz_chart(architecture)
