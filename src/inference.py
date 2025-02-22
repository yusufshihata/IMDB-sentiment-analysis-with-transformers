import torch

import torch

def predict(model, tokenizer, review, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Predicts the sentiment of a given text review using a trained Transformer model.

    Args:
        model (torch.nn.Module): The trained sentiment analysis model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing input text.
        review (str): The input text to analyze.
        device (str, optional): Device to run inference on ("cuda" or "cpu"). Defaults to GPU if available.

    Returns:
        str: The predicted sentiment ("Positive" or "Negative").
    """
    model.eval()  # Set model to evaluation mode
    model.to(device)

    # Tokenize input text (convert to tensor format)
    tokens = tokenizer(
        review, 
        padding="max_length", 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )

    # Move input tensors to the specified device
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # Forward pass (disable gradient computation for efficiency)
    with torch.no_grad():
        logits = model(input_ids)

    # Get the predicted class (0 = Negative, 1 = Positive)
    predicted_class = torch.argmax(logits, dim=1).item()

    # Convert class index to human-readable label
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    
    print(f"Review Sentiment: {sentiment}")

    return sentiment
