"""
NFTSF architecture with a shared GRU context encoder.

Trimmed from the source repo's `architecture_encoder.py`: this keeps only
the GRU encoder (documented there as the "Default encoder") and drops the
MLP/CNN/Transformer variants and the numbered `preset_stage1..16` ablation
presets, none of which this minimal repo's example uses.

Rationale
---------
The plain `create_nfm` (see `architecture.py`) feeds the raw (standardized)
x_past as `context` to EVERY AutoregressiveRationalQuadraticSpline layer.
Those K * len(hidden_layers_list) spline conditioners each re-encode the
past independently, and during sampling this re-encoding happens D times
sequentially (once per autoregressive dimension step).

This module adds a single shared encoder:

    standardized x_past  ->  encoder g(.)  ->  h  ->  flow conditioner
                             (runs ONCE)          (h is reused by every
                                                   spline layer and every
                                                   autoregressive step)

The flow still has O(D) sequential autoregressive sampling steps — that
is structural to A-RQS — but the per-step cost drops because the
conditioner input is compact and already processed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import normflows as nf


class _GRUEncoder(nn.Module):
    """Bidirectional GRU + small MLP head. Default encoder."""
    def __init__(self, past_dim: int, hidden: int, layers: int, context_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=past_dim, hidden_size=hidden,
            num_layers=layers, batch_first=True, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, context_dim),
        )

    def forward(self, x_past: torch.Tensor) -> torch.Tensor:
        if x_past.dim() == 2:
            x_past = x_past.unsqueeze(-1)
        _, h = self.gru(x_past)
        h_fwd = h[-2]
        h_bwd = h[-1]
        h_final = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.head(h_final)


class NFTSFEncoded(nn.Module):
    """Encoder + conditional RQS flow, trained end-to-end.

    Interface mirrors normflows.ConditionalNormalizingFlow but takes raw
    (standardized) x_past instead of a pre-computed context vector.
    """
    def __init__(self, encoder: nn.Module, flow: nf.ConditionalNormalizingFlow):
        super().__init__()
        self.encoder = encoder
        self.flow = flow

    def forward_kld(self, x_future: torch.Tensor, x_past: torch.Tensor) -> torch.Tensor:
        context = self.encoder(x_past)
        return self.flow.forward_kld(x_future, context=context)

    def log_prob(self, x_future: torch.Tensor, x_past: torch.Tensor) -> torch.Tensor:
        context = self.encoder(x_past)
        return self.flow.log_prob(x_future, context=context)

    @torch.no_grad()
    def sample(self, num_samples: int, x_past: torch.Tensor):
        context = self.encoder(x_past)
        return self.flow.sample(num_samples, context=context)


def create_nfm_encoder(
    device: torch.device,
    n_past: int,
    n_future: int,
    past_dim: int = 1,
    encoder_hidden: int = 128,
    encoder_layers: int = 2,
    context_dim: int = 64,
    K: int = 6,
    hidden_units: int = 64,
    hidden_layers_list=(1, 2),
    tail_bound: float = 30.0,
) -> NFTSFEncoded:
    """Create a GRU-encoder + conditional A-RQS flow, trained jointly.

    Defaults match the source repo's `preset_stage1` ("Encoder + FULL
    flow"), which is `create_nfm_encoder`'s own default configuration.
    """
    enc = _GRUEncoder(past_dim=past_dim, hidden=encoder_hidden,
                       layers=encoder_layers, context_dim=context_dim)

    prior = nf.distributions.DiagGaussian(n_future, trainable=False)
    flows = []
    for _ in range(K):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                n_future, hidden_layers, hidden_units,
                num_context_channels=context_dim, tail_bound=tail_bound,
            )
            for hidden_layers in hidden_layers_list
        ]
        flows += [nf.flows.LULinearPermute(n_future)]
    cflow = nf.ConditionalNormalizingFlow(q0=prior, flows=flows)

    return NFTSFEncoded(encoder=enc, flow=cflow).to(device)
