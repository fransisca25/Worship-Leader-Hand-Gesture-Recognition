import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        scaled += mask

    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, max_seq_len, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, max_seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, max_seq_len, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        residual_x = x
        x = self.attention(x, mask=None)
        x = self.dropout(x)
        x = self.norm(x + residual_x)

        residual_x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm(x + residual_x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x


class MultiheadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_layer = nn.Linear(d_model, d_model)
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, query, context, mask=None):
        batch_size, query_len, d_model = query.size()
        context_len = context.size(1)

        q = self.q_layer(query)
        kv = self.kv_layer(context)

        q = q.reshape(batch_size, query_len, self.num_heads, self.head_dim)
        kv = kv.reshape(batch_size, context_len, self.num_heads, 2 * self.head_dim)

        q = q.permute(0, 2, 1, 3)
        kv = kv.permute(0, 2, 1, 3)

        k, v = kv.chunk(2, dim=-1)

        values, attention = scaled_dot_product(q, k, v, mask)

        values = values.permute(0, 2, 1, 3).contiguous()
        values = values.reshape(batch_size, query_len, self.num_heads * self.head_dim)

        out = self.linear_layer(values)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiheadCrossAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, mask=None):
        residual_query = query

        query = self.cross_attn(query, context, mask=mask)
        query = self.dropout(query)
        query = self.norm(query + residual_query)

        residual_query = query

        query = self.ffn(query)
        query = self.dropout(query)
        query = self.norm(query + residual_query)

        return query


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        query, context = inputs

        for module in self._modules.values():
            query = module(query, context)

        return query


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, d_ff, num_heads, dropout)
                                          for _ in range(num_layers)])

    def forward(self, x, y):
        y = self.layers(x, y)
        return y


class HandGestureTransformer(nn.Module):
    def __init__(
            self,
            input_dim,
            d_model,
            d_ff,
            num_heads,
            dropout,
            num_layers,
            num_classes,
            max_seq_len
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model)
        )

        self.encoder = TransformerEncoder(d_model, d_ff, num_heads, dropout, num_layers)

        self.decoder = TransformerDecoder(d_model, d_ff, num_heads, dropout, num_layers)


        self.decoder_query = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()

        x = self.input_projection(x)
        x = x + self.pos_embedding[:, :seq_len, :]
        encoder_out = self.encoder(x)
        query = self.decoder_query.repeat(batch_size, 1, 1)
        decoder_out = self.decoder(query, encoder_out)
        decoder_out = decoder_out.squeeze(1)
        logits = self.classifier(decoder_out)
        return logits

