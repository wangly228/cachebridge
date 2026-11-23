"""Predeclared diagnostics for BF16, separate from exact cache invariants."""
import torch


PROTOCOL = dict(
    version="bf16-prefix-v1",
    bf16_epsilon=2**-7,
    max_abs_budget_eps=2,
    relative_l2_budget_eps=4,
    near_tie_regret_budget_multiplier=2,
    hard_checks=["identical teacher-forced input/position", "expected contiguous prefix hits",
                 "hash chain and tokens", "cached prefix bitwise unchanged",
                 "same-shape repeated logits bitwise equal", "real KV roundtrip bitwise equal",
                 "post-roundtrip logits bitwise equal"],
    numerical_scope="Engineering screening envelope, not a proven error bound or full baseline acceptance. "
                    "Known regression data is not independent calibration. Held-out cases are reported separately.",
    performance_scope="Diagnostic hooks synchronize and copy logits/KV; never use these times as performance results.",
)


def bitwise_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def compare_logits(reference, actual):
    if reference.shape != actual.shape or reference.ndim != 1:
        raise ValueError("compare one vocabulary row at the same position")
    a, b = reference.float(), actual.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return dict(passed=False, reason="nonfinite_logits")
    delta = (a - b).abs()
    scale = max(1.0, a.abs().max().item(), b.abs().max().item())
    budget = PROTOCOL["max_abs_budget_eps"] * PROTOCOL["bf16_epsilon"] * scale
    relative_l2 = (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a).clamp_min(1e-12)).item()
    ref_id, actual_id = a.argmax().item(), b.argmax().item()
    top2 = torch.topk(a, 2).values
    regret = (a.max() - a[actual_id]).item()
    max_abs = delta.max().item()
    return dict(
        passed=max_abs <= budget and relative_l2 <= PROTOCOL["relative_l2_budget_eps"] * PROTOCOL["bf16_epsilon"],
        max_abs=max_abs, max_abs_budget=budget, mean_abs=delta.mean().item(), relative_l2=relative_l2,
        reference_argmax=ref_id, actual_argmax=actual_id, argmax_equal=ref_id == actual_id,
        reference_top2_margin=(top2[0] - top2[1]).item(), reference_regret=regret,
        within_near_tie_budget=regret <= PROTOCOL["near_tie_regret_budget_multiplier"] * budget,
    )
