import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text)
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

# Extract URLs
def extract_urls(text):
    return re.findall(r'(https?://\S+)', text)

st.set_page_config(page_title="Email Phishing Detector", page_icon="📧")

st.title("📧 AI + ML Email Phishing Detection System")
st.markdown("### ⚡ Fast ML + 🤖 Optional AI Analysis")

st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload Email (.txt)", type=["txt"])

if uploaded_file:
    user_input = uploaded_file.read().decode("utf-8")
    st.text_area("📄 File Content", user_input, height=200)
else:
    user_input = st.text_area("✉️ Enter Email Message")

use_ai = st.checkbox("🤖 Enable AI Deep Analysis (slower)")

if st.button("🚀 Analyze Email"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter or upload a message")
    else:
        st.markdown("---")

        msg_clean = clean_text(user_input)
        msg_vec = vectorizer.transform([msg_clean])

        prediction = model.predict(msg_vec)
        prob = model.predict_proba(msg_vec)

        st.subheader("⚡ ML Prediction")

        if prediction[0] == "Phishing Email":
            st.error("🚨 This is a PHISHING Email")
        else:
            st.success("✅ This is a SAFE Email")

        st.write(f"Phishing: {prob[0][0]*100:.2f}%")
        st.write(f"Safe: {prob[0][1]*100:.2f}%")
        st.progress(int(max(prob[0]) * 100))

        urls = extract_urls(user_input)
        if urls:
            st.subheader("🔗 Links Found")
            for url in urls:
                st.write(url)

        if use_ai:
            with st.spinner("Running AI analysis..."):
                from transformers import pipeline
                classifier = pipeline("text-classification")
                result = classifier(user_input)[0]

                st.subheader("🤖 AI Analysis")
                st.write(result)

st.markdown("---")
st.write("Model Accuracy: ~95%")
st.info("💡 Tip: Try phishing messages like 'Click here to verify your bank account'")
