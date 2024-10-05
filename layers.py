import torch
import torch.nn as nn
import math
from utils import clones
from torch.nn.functional import log_softmax


class LayerNorm(nn.Module):
    "Construct a layernorm module - https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class SublayerConnection(nn.Module):
    """
    A residual connection (https://arxiv.org/abs/1512.03385) followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
    
def attention(query, key, value, mask=None, dropout=None):
    # Your code here
    d_k = query.size(-1)
    
    # Step 1: Compute the dot product of queries and keys, then scale by sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Step 2: Apply the mask (expand mask if needed to match dimensions), where mask == 0 set score to -inf
    if mask is not None:
        # Expand mask to match the dimensions of scores
        mask = mask.unsqueeze(1)  # shape (batch_size, 1, 1, seq_len_k)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Apply the softmax to the attention scores to get the attention weights
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)

    # Step 4: Apply dropout to the attention weights, if a dropout module is provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Step 5: Compute the final output by multiplying the attention weights with the values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h  # Dimension of each head

        # Ensure d_model is divisible by the number of heads
        assert d_model % h == 0, "d_model must be divisible by h"

        # Linear layers to project input query, key, and value to d_k for each head
        self.linear_query = nn.Linear(d_model, h * self.d_k)
        self.linear_key = nn.Linear(d_model, h * self.d_k)
        self.linear_value = nn.Linear(d_model, h * self.d_k)

        # Linear layer to project the concatenated attention output back to d_model
        self.linear_out = nn.Linear(h * self.d_k, d_model)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Scaled dot-product attention function
        self.attention = attention  # Use the attention function from the previous task

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Step 1: Project the query, key, and value using the linear layers
        query = self.linear_query(query)  # (batch_size, seq_len, h * d_k)
        key = self.linear_key(key)        # (batch_size, seq_len, h * d_k)
        value = self.linear_value(value)  # (batch_size, seq_len, h * d_k)

        # Step 2: Split the projections into multiple heads and reshape
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)      # (batch_size, h, seq_len, d_k)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)

        # Step 3: Apply attention to each head
        x, attn_weights = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        self.attn = attn_weights

        # Step 4: Concatenate the results of all heads and reshape
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)  # (batch_size, seq_len, h * d_k)

        # Step 5: Project the concatenated outputs using the final linear layer
        output = self.linear_out(x)

        return output
    
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

    

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())    

    