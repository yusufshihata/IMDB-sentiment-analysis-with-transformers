import streamlit as st
import torch
import torch.optim as optim
from transformers import AutoTokenizer
from src.model import TransformerEncoder  # Import your Transformer model class
from src.lr_scheduler import TransformerScheduler
from src.utils import load_checkpoint

# Load model and tokenizer
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")  # Adjust for your tokenizer

net = TransformerEncoder(vocab_size=30522, d_model=768, num_heads=12, num_layers=6, d_ff=3072, num_classes=2)
optimizer = optim.Adam(net.parameters(), lr=1e-4)

model = load_checkpoint(
    "./models/checkpoint_3.pth",
    net,
    optimizer,
    TransformerScheduler(optimizer, 768)
)
tokenizer = load_tokenizer()

# Streamlit UI
st.title("Sentiment Analysis App")
st.subheader("Enter a review and see if it's Positive or Negative!")

# User Input
review_text = st.text_area("Enter your review:", "")

if st.button("Analyze Sentiment"):
    if review_text.strip():
        # Tokenize and preprocess
        tokens = tokenizer(review_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt").unsqueeze(0)
        input_ids = tokens["input_ids"]
        
        # Make prediction
        with torch.no_grad():
            logits = model(input_ids)
        
        # Get sentiment
        sentiment = "Positive 😊" if torch.argmax(logits, dim=1).item() == 1 else "Negative 😞"
        
        # Display result
        st.success(f"**Sentiment:** {sentiment}")
    else:
        st.warning("Please enter a review first.")