from src.layers import SelfAttention, FeedForward, Embedding, PositionalEncoding
from torch import nn

class EncoderBlock(nn.Module):
    """Single Transformer Encoder Block."""
    def __init__(self, embed_dim, num_heads, hidden_dim = 2048):
        super().__init__()
        # Initialize the encoder block with self-attention and feed-forward layers
        self.attention = SelfAttention(embed_dim, num_heads)
        self.normalize1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)  # Optional dropout layer
        self.ffn = FeedForward(embed_dim, hidden_dim = 2048)
        self.normalize2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(0.1)  # Optional dropout layer

    def forward(self, X, return_attention=False):
        if return_attention:
            output_attention, scores = self.attention(X, return_attention=True)
        else:
            output_attention = self.attention(X, return_attention=False)
            scores = None 
        X = self.normalize1(X + self.dropout1(output_attention))
        # Wihtout dropout
        #X = self.normalize1(X + output_attention)
        output_ffn = self.ffn(X)
        X = self.normalize2(X + self.dropout2(output_ffn)) 
        # Without dropout
        #X = self.normalize2(X + output_ffn)
        return X, scores

class TransformerEncoder(nn.Module):
    """Stack of Transformer Encoder Blocks."""
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim = 2048):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, X, return_attention=False):
        attentions = []
        for layer in self.layers:
            X, scores = layer(X, return_attention=return_attention)
            if return_attention:
                attentions.append(scores)
        if return_attention:
            return X, attentions
        return X


class MaskedLanguageModel(nn.Module):
    """Masked Language Model with Transformer Encoder."""
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, vocab_size, max_length):
        super().__init__()
        # Initialize the model components
        self.embedding = Embedding(vocab_size, embed_dim)
        self.position = PositionalEncoding(max_length, embed_dim)
        self.transformer = TransformerEncoder(num_layers, embed_dim, num_heads, hidden_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        # Weight tying, ended up not using it
        # self.output.weight = self.Embedding.Embedding.weight
    
    def forward(self, X, return_attention):
        X = self.embedding(X)
        X = self.position(X)
        if return_attention:
            X, attention_scores = self.transformer(X, return_attention=True)
            return self.output(X), attention_scores
        else:
            X = self.transformer(X)
            return self.output(X)
