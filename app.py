import streamlit as st
import pickle
import os

model = pickle.load(open(os.path.join(os.path.dirname(__file__), "fake_news_model (1).pkl"), "rb"))
tfidf = pickle.load(open(os.path.join(os.path.dirname(__file__), "tfidf_vectorizer (1).pkl"), "rb"))

st.title("Fake News Detector")
st.write("Copy-Paste any news article below to check if it's real or fake!")

news = st.text_area("Enter news article here:", height=200)

if st.button("Check News"):
    if news.strip() == "":
        st.warning("⚠️ Please enter a news article first!")
    else:
        sample = tfidf.transform([news])
        prediction = model.predict(sample)
        if prediction[0] == 0:
            st.success("This appears to be REAL news!")
        else:
            st.error("This appears to be FAKE news!")
