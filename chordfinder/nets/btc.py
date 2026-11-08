"""BTC: Bi-directional Transformer for Chord recognition (inference only).

Adapted from https://github.com/jayg996/BTC-ISMIR19 (MIT License,
Copyright (c) 2019 Jonggwon Park): J. Park, K. Choi, S. Jeon, D. Kim, J. Park,
"A Bi-Directional Transformer for Musical Chord Recognition", ISMIR 2019.

Module and parameter names match the original so its checkpoints load
unchanged. Training-only code (losses, dropout) was removed.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

# Feature extraction settings the published checkpoints were trained with.
SAMPLE_RATE = 22050
N_BINS = 144
BINS_PER_OCTAVE = 24
HOP_LENGTH = 2048
CHUNK_SECONDS = 10.0
TIMESTEP = 108                          # frames per model input
FRAME_SECONDS = CHUNK_SECONDS / TIMESTEP  # the original code's time unit

CONFIG = dict(feature_size=144, timestep=TIMESTEP, num_layers=8, num_heads=4, hidden_size=128,
              total_key_depth=128, total_value_depth=128, filter_size=128)

ROOTS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_LARGE_QUALITIES = ("min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7",
                    "dim7", "hdim7", "sus2", "sus4")


def labels(num_chords: int) -> list[str]:
    """Class index → Harte-style chord label for the 25- or 170-class model."""
    if num_chords == 25:
        out = []
        for root in ROOTS:
            out += [root, f"{root}:min"]
        return out + ["N"]
    out = []
    for i in range(168):
        root, quality = ROOTS[i // 14], _LARGE_QUALITIES[i % 14]
        out.append(root if quality == "maj" else f"{root}:{quality}")
    return out + ["X", "N"]


def _bias_mask(length: int) -> torch.Tensor:
    mask = torch.triu(torch.full((length, length), float("-inf")), 1)
    return mask.unsqueeze(0).unsqueeze(1)


def _timing_signal(length: int, channels: int, min_timescale=1.0, max_timescale=1.0e4) -> torch.Tensor:
    position = np.arange(length)
    num_timescales = channels // 2
    increment = math.log(max_timescale / min_timescale) / (num_timescales - 1)
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float64) * -increment)
    scaled = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]])
    return torch.from_numpy(signal.reshape([1, length, channels])).float()


class LayerNorm(nn.Module):
    """The original's layer norm (unbiased std, eps added to the std)."""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, num_heads, bias_mask):
        super().__init__()
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

    def _split(self, x):
        b, t, d = x.shape
        return x.view(b, t, self.num_heads, d // self.num_heads).permute(0, 2, 1, 3)

    def forward(self, x):
        q = self._split(self.query_linear(x)) * self.query_scale
        k = self._split(self.key_linear(x))
        v = self._split(self.value_linear(x))
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        logits = logits + self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].to(logits)
        context = torch.matmul(torch.softmax(logits, dim=-1), v)
        b, h, t, d = context.shape
        context = context.permute(0, 2, 1, 3).contiguous().view(b, t, h * d)
        return self.output_linear(context)


class Conv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3):
        super().__init__()
        self.pad = nn.ConstantPad1d((kernel_size // 2, (kernel_size - 1) // 2), 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, x):
        return self.conv(self.pad(x.permute(0, 2, 1))).permute(0, 2, 1)


class PositionwiseFeedForward(nn.Module):
    """conv → ReLU → conv → ReLU (the original applies ReLU after both layers)."""

    def __init__(self, input_depth, filter_size, output_depth):
        super().__init__()
        self.layers = nn.ModuleList([Conv(input_depth, filter_size), Conv(filter_size, output_depth)])
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, bias_mask):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, bias_mask)
        self.positionwise_convolution = PositionwiseFeedForward(hidden_size, filter_size, hidden_size)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm_mha(x))
        return x + self.positionwise_convolution(self.layer_norm_ffn(x))


class BiDirectionalSelfAttention(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, max_length):
        super().__init__()
        forward_mask = _bias_mask(max_length)
        args = (hidden_size, total_key_depth, total_value_depth, filter_size, num_heads)
        self.attn_block = SelfAttentionBlock(*args, forward_mask)
        self.backward_attn_block = SelfAttentionBlock(*args, forward_mask.transpose(2, 3))
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        return self.linear(torch.cat((self.attn_block(x), self.backward_attn_block(x)), dim=2))


class BiDirectionalSelfAttentionLayers(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length):
        super().__init__()
        self.timing_signal = _timing_signal(max_length, hidden_size)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(*[
            BiDirectionalSelfAttention(hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                                       max_length) for _ in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, x):
        y = self.embedding_proj(x)
        y = y + self.timing_signal[:, :x.shape[1], :].to(y)
        return self.layer_norm(self.self_attn_layers(y))


class SoftmaxOutputLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.output_projection = nn.Linear(hidden_size, output_size)
        # Present in the checkpoints but unused by the softmax output.
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2, batch_first=True,
                            bidirectional=True)

    def forward(self, hidden):
        return self.output_projection(hidden)


class BTC(nn.Module):
    def __init__(self, num_chords: int):
        super().__init__()
        c = CONFIG
        self.timestep = c["timestep"]
        self.self_attn_layers = BiDirectionalSelfAttentionLayers(
            c["feature_size"], c["hidden_size"], c["num_layers"], c["num_heads"], c["total_key_depth"],
            c["total_value_depth"], c["filter_size"], c["timestep"])
        self.output_layer = SoftmaxOutputLayer(c["hidden_size"], num_chords)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, timestep, 144) normalised log-CQT → logits (batch, timestep, classes)."""
        return self.output_layer(self.self_attn_layers(features))


def load_checkpoint(path, num_chords: int, device: str = "cpu") -> tuple[BTC, float, float]:
    """Load a published BTC checkpoint. Returns (model, feature mean, feature std)."""
    import numpy

    # The checkpoints store mean/std as numpy scalars; allow exactly those types.
    scalar = getattr(numpy, "_core", numpy.core).multiarray.scalar
    safe = [(scalar, "numpy.core.multiarray.scalar"), numpy.dtype]
    safe += [type(numpy.dtype(t)) for t in ("float64", "float32")]
    with torch.serialization.safe_globals(safe):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    model = BTC(num_chords)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, float(ckpt["mean"]), float(ckpt["std"])
