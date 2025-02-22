import torch
from src.lr_scheduler import TransformerScheduler

import torch

def compute_accuracy(preds, labels):
    """
    Computes the accuracy of predictions against the true labels.

    Args:
        preds (torch.Tensor): Model output logits of shape (batch_size, num_classes).
        labels (torch.Tensor): Ground truth labels of shape (batch_size,).

    Returns:
        float: Accuracy as a value between 0 and 1.
    """
    # Get the predicted class (index of the highest logit)
    preds = torch.argmax(preds, dim=1)  
    
    # Count correct predictions
    correct = (preds == labels).sum().item()
    total = labels.size(0)

    # Compute accuracy as a fraction
    return correct / total


def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    """
    Saves the model, optimizer, and scheduler state to a file.

    Args:
        model (torch.nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): The current epoch number.
        filename (str): Path where the checkpoint should be saved.

    Returns:
        None
    """
    # Store model and optimizer state
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    
    # If scheduler is a custom TransformerScheduler, save additional parameters
    if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        checkpoint["scheduler_params"] = {
            "d_model": scheduler.d_model,
            "warmup_steps": scheduler.warmup_steps
        }

    # Save checkpoint file
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer, scheduler=None):
    """
    Loads a saved model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer): Optimizer to restore state.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to restore state.

    Returns:
        int: The last saved epoch number.
        torch.optim.lr_scheduler._LRScheduler (optional): Restored scheduler.
    """
    # Load checkpoint file
    checkpoint = torch.load(filename, map_location=torch.device("cpu"))
    
    # Restore model and optimizer state
    model.load_state_dict(checkpoint["model_state_dict"], map_location=torch.device("cpu"))
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # If a scheduler exists in the checkpoint, restore its parameters
    if scheduler and "scheduler_params" in checkpoint:
        scheduler = TransformerScheduler(optimizer, **checkpoint["scheduler_params"])
    
    # Return the last saved epoch and the scheduler (if restored)
    return checkpoint.get("epoch", 0), scheduler
