import os
import torch
import streamlit as st
from transformers import AutoTokenizer
from model import TransformerEncoder  # Load your custom model
from inference import predict  # Import prediction function

# Streamlit UI Setup
st.set_page_config(page_title="IMDB Sentiment Analysis with Transformers", layout="centered")
st.title("IMDB Sentiment Analysis")
st.write("Enter a movie review and get the predicted sentiment!")

# Model directory
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")  # Adjust the filename if needed

# Step 1: Load Tokenizer
st.info("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Change if needed
    st.success("Tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Failed to load tokenizer: {e}")
    st.stop()

# Step 2: Load the Model
st.info("Loading model...")
try:
    # Define model parameters (update based on your model architecture)
    model = TransformerEncoder(vocab_size=30522, d_model=768, num_heads=12, num_layers=6, d_ff=3072)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Step 3: Streamlit UI for Input
user_input = st.text_area("Enter a review:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict(model, tokenizer, user_input, device)
        st.subheader(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text before analyzing.")
