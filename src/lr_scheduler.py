import torch

class TransformerScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Custom learning rate scheduler for Transformer models based on the original paper:
    "Attention is All You Need" (Vaswani et al., 2017).
    
    This scheduler increases the learning rate linearly for the first `warmup_steps` steps 
    and then decays it proportionally to the inverse square root of the step number.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate needs scheduling.
        d_model (int): The model dimension size (hidden size) of the Transformer.
        warmup_steps (int, optional): The number of warmup steps. Default is 4000.
    
    Formula:
        lr = (d_model ** -0.5) * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)

    Example usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=4000)

        for step in range(1, num_training_steps):
            optimizer.step()
            scheduler.step()
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        # Define the learning rate schedule function
        def lr_lambda(step: int) -> float:
            """
            Compute the learning rate scaling factor.

            Args:
                step (int): Current training step.

            Returns:
                float: Scaling factor for learning rate.
            """
            step = max(1, step)  # Ensure step is at least 1 to avoid division by zero
            return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

        super().__init__(optimizer, lr_lambda)
