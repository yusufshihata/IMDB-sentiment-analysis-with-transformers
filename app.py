import asyncio
import os
import torch
import streamlit as st
from transformers import AutoTokenizer
from src.model import TransformerEncoder
from src.inference import predict

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")
st.title("IMDB Sentiment Analysis")
st.write("Enter a movie review and get the predicted sentiment!")

# Model path
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "checkpoint_3.pth")

# Load Tokenizer
st.info("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    st.success("Tokenizer loaded!")
except Exception as e:
    st.error(f"Failed to load tokenizer: {e}")
    st.stop()

# Load Model
st.info("Loading model...")
try:
    model = TransformerEncoder(vocab_size=30522, d_model=768, num_heads=12, num_layers=6, d_ff=3072)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    st.success("Model loaded!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# UI for Input
user_input = st.text_area("Enter a review:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict(model, tokenizer, user_input, device)
        st.subheader(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter text before analyzing.")
