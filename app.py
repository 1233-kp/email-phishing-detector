import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import pipeline

@st.cache_resource
def load_ml_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        nltk.download('stopwords', quiet=True)
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Files 'model.pkl' or 'vectorizer.pkl' not found.")
        return None, None

@st.cache_resource
def load_ai_pipeline():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

model, vectorizer = load_ml_assets()
ai_classifier = load_ai_pipeline()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

def extract_urls(text):
    return re.findall(r'(https?://\S+)', text)

st.set_page_config(page_title="Phishing Shield AI", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .medium-font { font-size:18px !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Phishing Shield: Advanced Detection System")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("📩 Paste Email Content:", height=250, placeholder="Check for suspicious activity...")
    uploaded_file = st.file_uploader("Or upload .txt", type=["txt"])
    if uploaded_file:
        user_input = uploaded_file.read().decode("utf-8")

with col2:
    st.markdown("### ⚙️ Analysis Settings")
    use_ai = st.toggle("🤖 Enable AI Linguistic Scan", value=True)
    st.info("AI detects 'Psychological Triggers' like fear or urgency.")

if st.button("🚀 RUN FULL DIAGNOSTIC", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please provide email text.")
    elif model is None:
        st.error("System Offline: ML Models missing.")
    else:
        msg_clean = clean_text(user_input)
        msg_vec = vectorizer.transform([msg_clean])
        prob = model.predict_proba(msg_vec)[0]
        base_risk = prob[1] 

        urls = extract_urls(user_input)
        technical_red_flags = 0.0
        
        trusted = ["google.com", "amazon.in", "amazon.com", "hdfcbank.com", "sbi.co.in", "paypal.com", "microsoft.com"]
        
        urgency_words = ["urgent", "immediately", "suspended", "action required", "within 24 hours"]
        if any(word in user_input.lower() for word in urgency_words):
            technical_red_flags += 0.1

        for url in urls:
            domain = url.split("//")[-1].split("/")[0].lower()
            is_domain_trusted = any(t in domain for t in trusted)
            
            if url.startswith("http://"):
                technical_red_flags += 0.4

            domain_keywords = ["secure", "verify", "update", "login", "alert", "security"]
            if not is_domain_trusted:
                if any(k in domain for k in domain_keywords):
                    technical_red_flags += 0.5 
            else:
                technical_red_flags -= 0.3 

        final_risk = max(0.0, min(base_risk + technical_red_flags, 1.0))

        st.markdown("## 📊 Diagnostic Report")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.subheader("⚡ Threat Assessment")
            if final_risk > 0.65:
                st.error(f"### 🚨 PHISHING DETECTED")
            elif final_risk > 0.35:
                st.warning(f"### ⚠️ SUSPICIOUS ACTIVITY")
            else:
                st.success(f"### ✅ APPARENTLY SAFE")
            
            st.write(f"**Calculated Risk Score:** `{final_risk*100:.1f}%`")
            st.progress(float(final_risk))

        with res_col2:
            st.subheader("🔗 Link Scan")
            if urls:
                for url in urls:
                    status = "✅ Trusted" if any(t in url.lower() for t in trusted) else "❓ Unknown"
                    st.write(f"{status}: `{url}`")
            else:
                st.write("No external links detected.")

        if use_ai:
            st.markdown("---")
            with st.spinner("🤖 AI evaluating phrasing..."):
                ai_result = ai_classifier(user_input[:512])[0]
                label = ai_result["label"]
                conf = ai_result["score"] * 100

                is_dangerous = (label == "NEGATIVE" or final_risk > 0.6)
                bg_color = "#f8d7da" if is_dangerous else "#d4edda"
                text_color = "#721c24" if is_dangerous else "#155724"
                verdict_title = "DANGEROUS / AGGRESSIVE" if is_dangerous else "NORMAL / SYSTEMIC"

                st.markdown(f"""
                    <div style="background-color:{bg_color}; padding:25px; border-radius:15px; border-left: 10px solid {text_color};">
                        <p style="color:{text_color}; margin-bottom:0;" class="medium-font">🤖 <b>AI Linguistic Verdict:</b></p>
                        <h2 style="color:{text_color}; margin-top:0;" class="big-font">{verdict_title} ({conf:.1f}% Match)</h2>
                        <p style="color:{text_color};">Analyzed psychological triggers and intent behind the message phrasing.</p>
                    </div>
                """, unsafe_allow_html=True)
