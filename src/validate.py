import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def validate(
    model: torch.nn.Module,
    validloader: DataLoader,
    criterion: torch.nn.Module,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple:
    """
    Evaluates the model on a validation dataset.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        validloader (DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function used to compute validation loss.
        device (str, optional): Device to run validation on ("cuda" or "cpu"). Defaults to automatically detecting CUDA.

    Returns:
        tuple: A tuple containing the average validation loss and validation accuracy.
    """
    
    # Move model to the specified device
    model.to(device)
    model.eval()  # Set model to evaluation mode

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        loop = tqdm(validloader, desc="Validating", leave=True)

        for batch in loop:
            # Load inputs and labels onto the specified device
            review, label = batch["review"].to(device), batch["label"].to(device)

            # Forward pass
            pred = model(review)

            # Compute loss
            loss = criterion(pred, label)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(pred, dim=1)  # Get predicted class index
            correct_predictions += (predicted == label).sum().item()
            total_predictions += label.size(0)

            loop.set_postfix(loss=loss.item())  # Update progress bar

    # Compute average loss and accuracy
    avg_loss = total_loss / len(validloader)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy
