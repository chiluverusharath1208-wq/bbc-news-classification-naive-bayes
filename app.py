import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("naive_bayes.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("BBC News Classification")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):

    transformed_text = vectorizer.transform([user_input])

    prediction = model.predict(transformed_text)

    st.success(f"Predicted Category: {prediction[0]}")