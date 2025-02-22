import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in the paper "Attention is All You Need".
    
    This module adds positional information to token embeddings using sinusoidal functions.
    The encoding is precomputed and stored in a buffer to avoid recomputation.

    Args:
        d_model (int): The dimensionality of the model (embedding size).
        max_len (int, optional): Maximum sequence length. Default is 512.
    
    Example:
        pe = PositionalEncoding(d_model=512)
        x = pe(torch.zeros(1, 10, 512))  # Add positional encoding to embeddings
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Save as buffer (not a parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Positionally encoded tensor of the same shape.
        """
        return x + self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    """
    Implements multi-head self-attention as described in the Transformer architecture.

    Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): Number of attention heads.

    Example:
        attn = MultiHeadSelfAttention(d_model=512, num_heads=8)
        x = torch.rand(2, 10, 512)  # Batch size 2, sequence length 10
        output = attn(x)  # Output shape (2, 10, 512)
    """

    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)  # Project input to Q, K, V
        self.out_proj = nn.Linear(d_model, d_model)  # Final projection layer
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Attention mask (if any).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.shape  # Batch size, sequence length, embedding dim
        qkv = self.qkv_proj(x).chunk(3, dim=-1)  # Split into Q, K, V
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        attn_scores = (q @ k.transpose(-2, -1)) / self.scale  # Scaled dot-product attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """
    Implements a position-wise feed-forward network (FFN) used in Transformer blocks.

    Args:
        d_model (int): The dimensionality of the model.
        d_ff (int): The hidden layer size in the feed-forward network.

    Example:
        ffn = FeedForward(d_model=512, d_ff=2048)
        x = torch.rand(2, 10, 512)
        output = ffn(x)  # Output shape (2, 10, 512)
    """

    def __init__(self, d_model: int, d_ff: int):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        return self.fc2(F.gelu(self.fc1(x)))  # GELU activation for non-linearity


class TransformerEncoderLayer(nn.Module):
    """
    Implements a single Transformer encoder block (Self-Attention + FFN + LayerNorm + Dropout).

    Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): Number of attention heads.
        d_ff (int): The hidden size of the feed-forward network.
        dropout (float, optional): Dropout rate. Default is 0.1.

    Example:
        layer = TransformerEncoderLayer(d_model=512, num_heads=8, d_ff=2048)
        x = torch.rand(2, 10, 512)
        output = layer(x)  # Output shape (2, 10, 512)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Transformer encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Attention mask.

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        x = x + self.dropout(self.self_attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """
    Implements a full Transformer encoder model for sentiment analysis.

    Args:
        vocab_size (int): Vocabulary size for embedding layer.
        d_model (int, optional): Model dimensionality. Default is 768.
        num_heads (int, optional): Number of attention heads. Default is 12.
        num_layers (int, optional): Number of Transformer encoder layers. Default is 6.
        d_ff (int, optional): Hidden size of the feed-forward network. Default is 3072.
        max_len (int, optional): Maximum sequence length. Default is 512.
        num_classes (int, optional): Number of output classes. Default is 2.

    Example:
        model = TransformerEncoder(vocab_size=30522)
        x = torch.randint(0, 30522, (2, 50))  # Batch size 2, seq length 50
        output = model(x)  # Output shape (2, num_classes)
    """

    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=6, d_ff=3072, max_len=512, num_classes=2):
        super(TransformerEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)  # Classification head

    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = self.norm(x).mean(dim=1)
        return self.fc(x)
