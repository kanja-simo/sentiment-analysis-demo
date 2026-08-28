import streamlit as st
import joblib

# Load the saved model and vectorizer (only runs once thanks to caching)
@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Page title and description
st.title("Sentiment Analysis Tool")
st.write("Enter a piece of text below and the model will predict whether it's positive or negative.")

# Text input box
user_text = st.text_area("Enter text here:", height=150)

# Only run prediction when the button is clicked
if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Convert the input text using the SAME vectorizer from training
        text_vector = vectorizer.transform([user_text])

        # Predict the label
        prediction = model.predict(text_vector)[0]

        # Get confidence scores for both classes
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities) * 100

        # Display the result
        if prediction == 'pos':
            st.success(f"Positive sentiment ({confidence:.1f}% confidence)")
        else:
            st.error(f"Negative sentiment ({confidence:.1f}% confidence)")