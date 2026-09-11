import torch
import torch.nn.functional as F


def action_token_kl(sighted_logits, blind_logits, labels, action_start_id, detach="blind"):
    """KL(p_sighted || q_blind) over the action-token vocab slice at trajectory positions.

    sighted_logits/blind_logits: [B, T, V]; labels: [B, T] (ignore_index -100).
    detach="blind": blind branch is a fixed target (fit the sharper camera-conditioned
    distribution inside the broader camera-blind one); "vision": the reverse.
    Returns a scalar; zero if the batch has no action-token positions.
    """
    shift_sel = labels[:, 1:] >= action_start_id
    if not shift_sel.any():
        return sighted_logits.new_zeros(())
    sighted = sighted_logits[:, :-1, action_start_id:][shift_sel].float()
    blind = blind_logits[:, :-1, action_start_id:][shift_sel].float()
    if detach == "blind":
        blind = blind.detach()
    elif detach == "vision":
        sighted = sighted.detach()
    else:
        raise ValueError(f"unknown detach mode: {detach}")
    log_p = F.log_softmax(sighted, dim=-1)
    log_q = F.log_softmax(blind, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1).mean()


def coupling_step_active(step, prob):
    """Deterministic step-modulo gate (all ranks stay on the same collective path)."""
    if prob <= 0.0:
        return False
    return step % max(1, round(1.0 / prob)) == 0
