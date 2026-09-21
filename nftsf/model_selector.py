"""
Automatic architecture selection.

The source repo (nftsf_ssh) exposes both architectures behind a manual
`--model_variant` CLI flag. This minimal repo replaces that manual choice
with one automatic rule, applied identically by both train.py and test.py:

    n_past > ENCODER_THRESHOLD  ->  GRU-encoder architecture (architecture_encoder.py)
    n_past <= ENCODER_THRESHOLD ->  plain (non-encoder) architecture (architecture.py)

Note: exactly ENCODER_THRESHOLD (64) uses the NON-encoder path, since the
rule is a strict "greater than" comparison.
"""

from nftsf.architecture import create_nfm
from nftsf.architecture_encoder import create_nfm_encoder

ENCODER_THRESHOLD = 64

ENCODER = "encoder"
NON_ENCODER = "non_encoder"


def select_variant(n_past: int) -> str:
    """Return which architecture n_past selects, per ENCODER_THRESHOLD."""
    if n_past > ENCODER_THRESHOLD:
        return ENCODER
    return NON_ENCODER


def build_model(device, n_past: int, n_future: int, past_dim: int = 1):
    """Construct the model automatically selected by n_past.

    Both architectures expose the same call signature afterward —
    model.log_prob(x_future, x_past) and model.sample(n, x_past) — so
    callers never need to branch on the variant once the model is built.

    Returns:
        model, variant (one of ENCODER / NON_ENCODER)
    """
    variant = select_variant(n_past)
    if variant == ENCODER:
        print(f"n_past={n_past} > {ENCODER_THRESHOLD} -> using GRU-encoder architecture")
        model = create_nfm_encoder(device, n_past=n_past, n_future=n_future, past_dim=past_dim)
    else:
        print(f"n_past={n_past} <= {ENCODER_THRESHOLD} -> using non-encoder architecture")
        model = create_nfm(device, latent_size=n_future, context_size=n_past * past_dim)
    return model, variant
