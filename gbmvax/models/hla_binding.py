"""
hla_binding.py — multi-task transformer for peptide–HLA binding.

Architecture (one model, two heads):

    peptide tokens  ───┐
                       │  ┌─────────────────────────┐
                       ├──▶│  shared encoder         │──▶ pooled embedding ──┬──▶ affinity head ──▶ log10(IC50 nM)
                       │  │  (concat + transformer) │                       │
    HLA pseudoseq   ───┘  └─────────────────────────┘                       └──▶ presentation head ──▶ P(presented)

Why this design:

1. Concat-then-encode (vs cross-attention): we treat the input as one
   sequence [PEP_TOKENS, SEP, HLA_TOKENS] with learned segment embeddings.
   This lets every peptide position attend directly to every HLA position
   AND vice versa, which is biologically appropriate — anchor residues in
   the peptide make hydrogen bonds with specific HLA groove residues, and
   we want the model to model that interaction. Cross-attention would
   work too but doubles the parameters; the concat-encoder is more
   parameter-efficient for short sequences (9 + 34 = 43 positions).

2. Multi-task with two heads: binding affinity (continuous, log-IC50) and
   presentation (binary, mass-spec). Affinity dataset is much larger
   (~hundreds of thousands of measurements) but biased toward what people
   chose to assay. Presentation dataset is smaller but reflects the real
   in vivo outcome. Joint training transfers signal between heads.

3. Custom segment embeddings instead of [CLS]/[SEP] alone: token 0..10 are
   peptide, token 11 is separator, tokens 12..45 are HLA. We add a learned
   segment embedding so the model can tell peptide from HLA positions even
   when amino acid identity is identical.

4. We POOL with mean-over-peptide-tokens (not [CLS] from BERT) because the
   binding signal is distributed across all peptide residues. Empirically
   this beats CLS pooling for peptide–MHC tasks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Sinusoidal positional encoding. Standard "Attention Is All You Need" form.
# We use this rather than learned positional embeddings because peptide
# positions have strong physical meaning (P1, P2 anchors, P-omega anchor)
# that benefits from a smooth, periodic encoding rather than independent
# per-position vectors.
# ----------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 64):
        super().__init__()
        # Build the [max_len, embed_dim] PE matrix once; register as buffer
        # (moves with .to(device), not a learnable parameter).
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))                          # [1, max_len, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, embed_dim]. Add positions in-place.
        return x + self.pe[:, : x.size(1), :]


# ----------------------------------------------------------------------------
# Main model.
# ----------------------------------------------------------------------------
class HLABindingTransformer(nn.Module):
    """
    Multi-task peptide–HLA binding model.

    Input:
        peptide_tokens: [batch, max_peptide_len]   long, padded with 0
        hla_tokens:     [batch, pseudoseq_len]     long
        peptide_mask:   [batch, max_peptide_len]   bool, True where pad
        hla_mask:       [batch, pseudoseq_len]     bool, True where pad (rare)

    Output:
        log_affinity:   [batch] — predicted log10(IC50 nM); lower = tighter binder
        presentation:   [batch] — logit; sigmoid -> P(presented)
    """

    def __init__(
        self,
        vocab_size: int = 21,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_hidden: int = 512,
        dropout: float = 0.1,
        max_peptide_len: int = 11,
        pseudoseq_len: int = 34,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.max_peptide_len = max_peptide_len
        self.pseudoseq_len = pseudoseq_len
        self.pad_token_id = pad_token_id
        # Total sequence length: peptide + separator + HLA.
        self.total_len = max_peptide_len + 1 + pseudoseq_len

        # Shared amino acid embedding. Index 0 = pad (we pre-allocate it
        # with a fixed zero embedding via the embedding init below; we also
        # mask it from attention so the choice of values doesn't matter
        # numerically — only by convention).
        self.aa_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

        # Segment embedding — tells the model which positions belong to
        # peptide (0), separator (1), HLA (2). Crucial because token vocab
        # is shared between peptide and HLA.
        self.segment_embed = nn.Embedding(3, embed_dim)

        # Positional encoding spans the full concatenated sequence.
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=self.total_len)

        # Standard transformer encoder. batch_first=True makes shapes
        # easier to reason about throughout.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            batch_first=True,
            activation="gelu",                 # GELU outperforms ReLU on small sequence tasks
            norm_first=True,                   # Pre-LayerNorm = more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads. Two-layer MLPs with GELU.
        # Affinity head: predicts log10(IC50) — unbounded scalar.
        self.affinity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        # Presentation head: predicts logit of P(presented).
        self.presentation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        # Learned separator token embedding — a single vector, broadcast
        # to every sample. Acts as a delimiter the model can attend to.
        self.sep_embedding = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers, normal init for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        # Force pad index to zero — the padding_idx kwarg on nn.Embedding
        # already does this, but we make it explicit.
        with torch.no_grad():
            self.aa_embed.weight[self.pad_token_id].fill_(0)

    def forward(
        self,
        peptide_tokens: torch.Tensor,            # [B, max_peptide_len]
        hla_tokens: torch.Tensor,                # [B, pseudoseq_len]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = peptide_tokens.size(0)
        device = peptide_tokens.device

        # ----- Embed peptide and HLA tokens -----
        pep_emb = self.aa_embed(peptide_tokens)          # [B, P, D]
        hla_emb = self.aa_embed(hla_tokens)              # [B, H, D]

        # ----- Add segment embeddings -----
        pep_seg = self.segment_embed(torch.zeros(B, self.max_peptide_len, dtype=torch.long, device=device))
        sep_seg = self.segment_embed(torch.ones(B, 1, dtype=torch.long, device=device))
        hla_seg = self.segment_embed(2 * torch.ones(B, self.pseudoseq_len, dtype=torch.long, device=device))

        pep_emb = pep_emb + pep_seg
        hla_emb = hla_emb + hla_seg
        sep_emb = self.sep_embedding.expand(B, -1, -1) + sep_seg          # [B, 1, D]

        # ----- Concatenate peptide + sep + HLA -----
        x = torch.cat([pep_emb, sep_emb, hla_emb], dim=1)                  # [B, total_len, D]
        x = self.pos_enc(x)

        # ----- Build attention mask: True at pad positions -----
        # nn.TransformerEncoder uses src_key_padding_mask where True = ignore.
        pep_pad = (peptide_tokens == self.pad_token_id)                    # [B, P]
        sep_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)       # sep is never padded
        hla_pad = (hla_tokens == self.pad_token_id)                        # [B, H]
        key_padding_mask = torch.cat([pep_pad, sep_pad, hla_pad], dim=1)   # [B, total_len]

        # ----- Encode -----
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)         # [B, total_len, D]

        # ----- Pool over peptide tokens only -----
        # Mean over real (non-pad) peptide positions. The HLA tokens have
        # already informed the peptide embeddings via attention; we don't
        # need to pool over them again.
        pep_out = x[:, : self.max_peptide_len, :]                          # [B, P, D]
        pep_real_mask = (~pep_pad).float().unsqueeze(-1)                   # [B, P, 1], 1.0 where real
        pooled = (pep_out * pep_real_mask).sum(dim=1) / pep_real_mask.sum(dim=1).clamp(min=1.0)  # [B, D]

        # ----- Heads -----
        log_affinity = self.affinity_head(pooled).squeeze(-1)              # [B]
        presentation = self.presentation_head(pooled).squeeze(-1)          # [B]

        return log_affinity, presentation


def multitask_loss(
    log_aff_pred: torch.Tensor,
    log_aff_true: torch.Tensor,
    aff_mask: torch.Tensor,                       # 1.0 where this row has an affinity label
    pres_pred: torch.Tensor,
    pres_true: torch.Tensor,
    pres_mask: torch.Tensor,                      # 1.0 where this row has a presentation label
    binding_weight: float = 1.0,
    presentation_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Multi-task loss: MSE on log-IC50 + BCE-with-logits on presentation.

    Each row contributes to only the head(s) it has labels for. The masks
    are how we handle the fact that affinity-only rows have no presentation
    label and vice versa. We sum (not average) inside each head and divide
    by the count of labeled rows — this prevents the loss magnitude from
    fluctuating with batch composition.
    """
    # Affinity loss (MSE on log-IC50). Squared error per row, masked, normalized.
    aff_sqerr = (log_aff_pred - log_aff_true) ** 2 * aff_mask
    n_aff = aff_mask.sum().clamp(min=1.0)
    aff_loss = aff_sqerr.sum() / n_aff

    # Presentation loss (BCE with logits). Per-row binary cross entropy,
    # masked, normalized. We use BCEWithLogitsLoss(reduction='none') to get
    # per-row losses, then mask.
    pres_per_row = F.binary_cross_entropy_with_logits(
        pres_pred, pres_true, reduction="none"
    )
    pres_loss = (pres_per_row * pres_mask).sum() / pres_mask.sum().clamp(min=1.0)

    total = binding_weight * aff_loss + presentation_weight * pres_loss

    return total, {
        "loss_total": float(total.detach().item()),
        "loss_affinity": float(aff_loss.detach().item()),
        "loss_presentation": float(pres_loss.detach().item()),
        "n_aff": float(n_aff.item()),
        "n_pres": float(pres_mask.sum().item()),
    }
