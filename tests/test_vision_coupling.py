import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.coupling_utils import action_token_kl, coupling_step_active

ASID = 10  # small action_start_id for a toy vocab of size 12 (2 action tokens)


def _toy(batch=1, t=4, vocab=12):
    g = torch.Generator().manual_seed(0)
    sighted = torch.randn(batch, t, vocab, generator=g)
    blind = torch.randn(batch, t, vocab, generator=g)
    labels = torch.full((batch, t), -100, dtype=torch.long)
    return sighted, blind, labels


def test_kl_zero_for_identical_logits():
    sighted, _, labels = _toy()
    labels[0, 2] = ASID  # one action position (predicted from logits at t=1)
    kl = action_token_kl(sighted, sighted.clone(), labels, ASID)
    assert torch.isclose(kl, torch.zeros(()), atol=1e-6)


def test_kl_matches_hand_value():
    sighted, blind, labels = _toy()
    labels[0, 2] = ASID
    # hand-compute over the 2-token slice at shifted position 1
    p = torch.softmax(sighted[0, 1, ASID:].double(), -1)
    q = torch.softmax(blind[0, 1, ASID:].double(), -1)
    expected = (p * (p.log() - q.log())).sum().item()
    kl = action_token_kl(sighted, blind, labels, ASID)
    assert math.isclose(kl.item(), expected, rel_tol=1e-5)


def test_detach_direction_blocks_gradient():
    sighted, blind, labels = _toy()
    labels[0, 2] = ASID
    s = sighted.clone().requires_grad_(True)
    b = blind.clone().requires_grad_(True)
    action_token_kl(s, b, labels, ASID, detach="blind").backward()
    assert b.grad is None or torch.all(b.grad == 0)
    assert s.grad is not None and s.grad.abs().sum() > 0
    s2 = sighted.clone().requires_grad_(True)
    b2 = blind.clone().requires_grad_(True)
    action_token_kl(s2, b2, labels, ASID, detach="vision").backward()
    assert s2.grad is None or torch.all(s2.grad == 0)
    assert b2.grad is not None and b2.grad.abs().sum() > 0


def test_non_action_positions_ignored():
    sighted, blind, labels = _toy()
    labels[0, 2] = ASID
    base = action_token_kl(sighted, blind, labels, ASID)
    # perturb logits at non-action positions and below the vocab slice: no effect
    sighted2 = sighted.clone()
    sighted2[0, 0, :] += 5.0        # non-action shifted position
    sighted2[0, 1, :ASID] += 5.0    # action position, but below the action vocab slice
    labels2 = labels.clone()
    labels2[0, 3] = 3               # a non-action (text) label must not join the mask
    kl = action_token_kl(sighted2, blind, labels2, ASID)
    assert torch.isclose(kl, base, atol=1e-6)


def test_all_masked_returns_zero():
    sighted, blind, labels = _toy()
    kl = action_token_kl(sighted, blind, labels, ASID)
    assert kl.item() == 0.0


def test_step_modulo_gating():
    assert not any(coupling_step_active(s, 0.0) for s in range(8))
    assert all(coupling_step_active(s, 1.0) for s in range(8))
    fires = [coupling_step_active(s, 0.25) for s in range(8)]
    assert fires == [True, False, False, False, True, False, False, False]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
