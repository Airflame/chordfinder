"""S-KEY key detector (inference only).

Adapted from https://github.com/deezer/skey (MIT License, Copyright (c)
2019-present Deezer SA): Y. Kong, G. Meseguer-Brocal, V. Lostanlen,
M. Lagrange, R. Hennequin, "S-KEY: Self-supervised Learning of Major and Minor
Keys from Audio", ICASSP 2025. The ConvNeXt blocks are originally from Meta
FAIR (https://github.com/facebookresearch/ConvNeXt).

Module and parameter names match the original so its checkpoint loads
unchanged.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# Output order of the classifier, as in the original key_map.
KEY_NAMES = ("A major", "Bb major", "B major", "C major", "C# major", "D major", "D# major", "E major",
             "F major", "F# major", "G major", "G# major", "B minor", "C minor", "C# minor", "D minor",
             "D# minor", "E minor", "F minor", "F# minor", "G minor", "G# minor", "A minor", "Bb minor")


def _vqt_class():
    import nnAudio.features.vqt

    class VQT(nnAudio.features.vqt.VQT):
        """Harmonic VQT stack (here with the single harmonic the checkpoint uses)."""

        def __init__(self, *, harmonics, fmin, n_bins, bins_per_octave=12, **kwargs):
            self.harmonics = harmonics
            self.bin_shifts = [round(bins_per_octave * math.log2(h)) for h in harmonics]
            self.n_bins_per_slice = n_bins
            low_octave_shift = min([0] + self.bin_shifts) / bins_per_octave
            fmin = fmin * (2 ** low_octave_shift)
            n_bins = n_bins + max([0] + self.bin_shifts) - min([0] + self.bin_shifts)
            super().__init__(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, verbose=False, **kwargs)

        def forward(self, x, output_format="Magnitude", normalization_type="librosa"):
            import torchaudio
            vqt = super().forward(x, output_format, normalization_type)
            slices = []
            for shift in self.bin_shifts:
                start = shift - min([0] + self.bin_shifts)
                slices.append(vqt[:, start:start + self.n_bins_per_slice, ...])
            hvqt = torch.stack(slices, dim=1)
            return (1.0 / 80.0) * torchaudio.transforms.AmplitudeToDB(top_db=80)(hvqt) + 1.0

    return VQT


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, layer_scale_init_value=1e-1):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                groups=in_channels, padding_mode="replicate")
        self.pwconv1 = nn.Linear(out_channels, 4 * out_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * out_channels, in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(out_channels))

    def forward(self, x):
        y = self.dwconv(x)
        y = nn.functional.layer_norm(y, y.shape[1:])
        y = y.permute(0, 2, 3, 1)
        y = self.gamma * self.pwconv2(self.act(self.pwconv1(y)))
        return x + y.permute(0, 3, 1, 2)


class TimeDownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 2), stride=(1, 2))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(nn.functional.layer_norm(x, x.shape[1:])))


class ChromaNet(nn.Module):
    def __init__(self, n_harmonics=1, out_channels=(2, 3, 40, 40, 30, 10, 3), kernels=(7, 7, 7, 7, 7, 5, 5)):
        super().__init__()
        in_channel = n_harmonics
        downsampling, blocks = [], []
        for out_channel, kernel in zip(out_channels, kernels):
            downsampling.append(TimeDownsamplingBlock(in_channel, out_channel))
            blocks.append(ConvNeXtBlock(out_channel, out_channel, kernel_size=kernel, padding=kernel // 2))
            in_channel = out_channel
        self.convnext_blocks = nn.ModuleList(blocks)
        self.time_downsampling_blocks = nn.ModuleList(downsampling)
        self.global_average_pool = nn.AdaptiveAvgPool2d((12, 1))
        self.classifier = nn.Conv2d(in_channel, 2, kernel_size=(1, 1))
        self.batch_norm = nn.BatchNorm2d(2, affine=False)

    def forward(self, x):
        for block, down in zip(self.convnext_blocks, self.time_downsampling_blocks):
            x = block(down(x))
        b, c, h, w = x.shape
        x = x.view(b, c, h // 12, 12, w).mean(dim=2)       # fold octaves into a chromagram
        x = self.classifier(self.global_average_pool(x))
        x = self.batch_norm(x).flatten(1)
        return torch.softmax(x, dim=-1)                     # (batch, 24)


class SKey(nn.Module):
    N_BINS = 84  # the crop the classifier sees (7 octaves)

    def __init__(self):
        super().__init__()
        self.hcqt = _vqt_class()(harmonics=[1], fmin=27.5, n_bins=99)
        self.chromanet = ChromaNet()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (samples,) mono at the checkpoint's sample rate → key probabilities (24,)."""
        spec = self.hcqt(audio.unsqueeze(0).unsqueeze(0) if audio.ndim == 1 else audio)
        spec = spec[:, :, :self.N_BINS, :]
        return self.chromanet(spec)[0]


def load_checkpoint(path, device: str = "cpu") -> tuple[SKey, int]:
    """Load the published S-KEY checkpoint. Returns (model, sample rate)."""
    import numpy

    scalar = getattr(numpy, "_core", numpy.core).multiarray.scalar
    safe = [(scalar, "numpy.core.multiarray.scalar"), numpy.dtype]
    safe += [type(numpy.dtype(t)) for t in ("float64", "float32", "int64")]
    with torch.serialization.safe_globals(safe):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    model = SKey()
    state = ckpt["stone"]
    model.hcqt.load_state_dict({k[len("hcqt."):]: v for k, v in state.items() if k.startswith("hcqt.")})
    model.chromanet.load_state_dict({k[len("chromanet."):]: v for k, v in state.items()
                                     if k.startswith("chromanet.")})
    model.to(device).eval()
    return model, int(ckpt["audio"]["sr"])
