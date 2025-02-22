import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_checkpoint
from validate import validate
from visualize import plot_metrics

def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    trainloader: DataLoader,
    validloader: DataLoader,
    epochs: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """
    Trains a given model using the provided optimizer, loss function, and data loaders.

    Args:
        model (nn.Module): The neural network model to train.
        criterion (nn.Module): Loss function for training.
        optimizer (optim.Optimizer): Optimization algorithm.
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        trainloader (DataLoader): DataLoader for the training dataset.
        validloader (DataLoader): DataLoader for the validation dataset.
        epochs (int): Number of training epochs.
        device (str, optional): Device to use for training ("cuda" or "cpu"). Defaults to "cuda" if available.

    Returns:
        None
    """
    model.to(device)
    model.train()

    total_losses = []  # Track validation losses
    total_accs = []  # Track validation accuracies
    best_accuracy = 0.0

    for epoch in range(epochs):
        loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        total_loss = 0.0

        for batch in loop:
            review, label = batch["review"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            pred = model(review)
            loss = criterion(pred, label)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Compute average training loss
        avg_train_loss = total_loss / len(trainloader)
        
        # Validate model performance
        avg_val_loss, val_accuracy = validate(model, validloader, criterion, device)
        total_losses.append(avg_val_loss)
        total_accs.append(val_accuracy)

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}"
        )

        # Save model checkpoint if performance improves
        if val_accuracy > best_accuracy:
            save_checkpoint(model, optimizer, scheduler, epoch, filename=f"checkpoint_{epoch}.pth")
            best_accuracy = val_accuracy

    # Plot training metrics after all epochs
    plot_metrics(total_losses, total_accs)