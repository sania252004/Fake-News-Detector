import streamlit as st
import pickle
import os

base_path = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_path, "fake_news_model.pkl"), "rb"))
tfidf = pickle.load(open(os.path.join(base_path, "tfidf_vectorizer.pkl"), "rb"))

st.title("🔍 Fake News Detector")
st.write("cCopy-Paste any news article below to check if it's real or fake!")
st.info("ℹ️ This model is optimized for political news articles.")

news = st.text_area("Enter news article here:", height=200)

if st.button("Check News"):
    if news.strip() == "":
        st.warning("⚠️ Please enter a news article first!")
    else:
        sample = tfidf.transform([news])
        prediction = model.predict(sample)
        if prediction[0] == 0:
            st.success("✅ This appears to be REAL news!")
        else:
            st.error("🚨 This appears to be FAKE news!")
