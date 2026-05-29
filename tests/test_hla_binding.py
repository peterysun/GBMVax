"""Tests for gbmvax.models.hla_binding."""

import torch

from gbmvax.models.hla_binding import HLABindingTransformer, multitask_loss


def test_model_forward_shape():
    """Forward returns (log_aff, presentation) with batch shape preserved."""
    model = HLABindingTransformer(
        embed_dim=64, num_heads=4, num_layers=2,
        max_peptide_len=11, pseudoseq_len=34,
    )
    B = 4
    pep = torch.randint(0, 21, (B, 11))
    hla = torch.randint(1, 21, (B, 34))         # No pads in HLA
    log_aff, pres = model(pep, hla)
    assert log_aff.shape == (B,)
    assert pres.shape == (B,)


def test_model_handles_padded_peptide():
    """Pad tokens should not break attention; output is finite."""
    model = HLABindingTransformer(
        embed_dim=64, num_heads=4, num_layers=2,
        max_peptide_len=11, pseudoseq_len=34,
    )
    pep = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0]])   # 9-mer + 2 pads
    hla = torch.randint(1, 21, (1, 34))
    log_aff, pres = model(pep, hla)
    assert torch.isfinite(log_aff).all()
    assert torch.isfinite(pres).all()


def test_multitask_loss_masks_correctly():
    """A row with affinity_mask=0 should not contribute to affinity loss."""
    # Two rows: row 0 has affinity only, row 1 has presentation only.
    log_aff_pred = torch.tensor([2.0, 3.0])
    log_aff_true = torch.tensor([2.0, 0.0])         # row 1 placeholder
    aff_mask = torch.tensor([1.0, 0.0])
    pres_pred = torch.tensor([0.0, 5.0])
    pres_true = torch.tensor([0.0, 1.0])
    pres_mask = torch.tensor([0.0, 1.0])

    loss, parts = multitask_loss(
        log_aff_pred, log_aff_true, aff_mask,
        pres_pred, pres_true, pres_mask,
    )
    # Affinity loss should be (2.0 - 2.0)**2 = 0 since only row 0 contributes.
    assert parts["loss_affinity"] == 0.0
    # Presentation loss should be > 0 because row 1's prediction is high
    # confidence and the label is 1 — but loss is non-trivial here.
    assert parts["loss_presentation"] > 0.0
    assert parts["n_aff"] == 1.0
    assert parts["n_pres"] == 1.0


def test_backward_pass():
    """Gradients flow through the full model."""
    model = HLABindingTransformer(
        embed_dim=32, num_heads=2, num_layers=1,
        max_peptide_len=11, pseudoseq_len=34,
    )
    pep = torch.randint(1, 21, (2, 11))
    hla = torch.randint(1, 21, (2, 34))
    log_aff, pres = model(pep, hla)
    loss, _ = multitask_loss(
        log_aff, torch.zeros(2), torch.ones(2),
        pres, torch.zeros(2), torch.ones(2),
    )
    loss.backward()
    # Some parameter must have a non-zero gradient.
    has_grad = any(
        (p.grad is not None) and (p.grad.abs().sum().item() > 0)
        for p in model.parameters()
    )
    assert has_grad
