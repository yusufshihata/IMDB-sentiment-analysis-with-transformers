import matplotlib.pyplot as plt

def plot_metrics(val_losses, val_accuracies):
    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)  # Create subplot for loss
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Progress')
    
    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)  # Create subplot for accuracy
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Progress')
    
    # Show the plots
    plt.tight_layout()
    plt.show()
