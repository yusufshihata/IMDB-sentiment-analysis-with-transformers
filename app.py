import os
import torch
import subprocess
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

# Streamlit UI Setup
st.set_page_config(page_title="Sentiment Analysis with Transformers", layout="centered")

st.title("IMDB Sentiment Analysis")
st.write("Enter a movie review and get the predicted sentiment!")

# Step 1: Download the model from Kaggle
model_dir = "./model"
if not os.path.exists(model_dir) or len(os.listdir(model_dir)) == 0:
    st.info("Downloading model from Kaggle... (This may take a few moments)")
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        subprocess.run([
            "kaggle", "kernels", "output", "yusufshihata20069/sentiment-analysis-with-transformers",
            "-p", model_dir
        ], check=True)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

# Step 2: Load the model
st.info("Loading model...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Step 3: Prediction Function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move to GPU if available
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    sentiment = torch.argmax(scores).item()
    return "Positive" if sentiment == 1 else "Negative"

# Step 4: Streamlit UI
user_input = st.text_area("Enter a review:", "")
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.subheader(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text before analyzing.")
