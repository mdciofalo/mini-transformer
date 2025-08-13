from torch import nn
import torch
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Self-Attention mechanism."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # Initialize parameters for self-attention
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim  // num_heads

        self.W_q  = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, X, return_attention=False):
        batch_size, seq_len, _ = X.shape

        # Reshape and transpose for multi-head attention
        Q = self.W_q(X).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(X).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(X).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores and apply softmax
        scores = torch.matmul(Q,torch.transpose(K,-2, -1)) / math.sqrt(self.head_dim)
        scores = torch.softmax(scores, dim=-1)

        # Compute the attention output
        attention = torch.matmul(scores, V)
        attention = attention.transpose(1, 2)

        # Combine heads and project back to original dimension
        combined = attention.reshape(batch_size, seq_len, self.embed_dim)
        output = self.W_out(combined)

        # Return attention scores if requested
        if return_attention:
            return output, scores
        return output

class FeedForward(nn.Module):
    """Feed-forward network used in Transformer blocks."""
    def __init__(self, embed_dim, hidden_dim = 2048): # Attention is All You Need paper set hidden dimensions to 2048 
        super().__init__()
        # Initialize the feed-forward network
        self.model = nn.Sequential(
            nn.Linear(embed_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, X):
        return self.model(X)
    
class Embedding(nn.Module):
    """Embedding layer for token IDs."""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Used this initially, but switched to nn.Embedding
        # self.embedding = nn.Parameter(torch.randn(vocab_size, embed_dim)) #
    
    def forward(self, X):
        return self.embedding(X)

class PositionalEncoding(nn.Module):
    """Positional encoding to add sequence information."""
    def __init__(self, max_length,embed_dim):
        super().__init__()
        # Initialize positional encoding parameters
        self.position = nn.Parameter(torch.randn(max_length, embed_dim))
    
    def forward(self, X):
        seq_len = X.size(1)
        return X + self.position[:seq_len]
