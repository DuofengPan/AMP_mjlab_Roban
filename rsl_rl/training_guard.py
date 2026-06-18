"""Training stability guards: detect PPO/AMP numerical blow-ups and abort early."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def total_mini_batches(num_learning_epochs: int, num_mini_batches: int) -> int:
    return int(num_learning_epochs) * int(num_mini_batches)


def summarize_returns(returns_batch: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        finite = returns_batch[torch.isfinite(returns_batch)]
        if finite.numel() == 0:
            return {
                "returns_min": float("nan"),
                "returns_max": float("nan"),
                "returns_abs_max": float("nan"),
            }
        return {
            "returns_min": float(finite.min().item()),
            "returns_max": float(finite.max().item()),
            "returns_abs_max": float(finite.abs().max().item()),
        }


def policy_std_snapshot(policy) -> dict[str, float]:
    with torch.no_grad():
        if getattr(policy, "noise_std_type", None) == "scalar" and hasattr(policy, "std"):
            std = policy.std.detach()
            if std.numel() == 0:
                return {"policy_std_min": float("nan"), "policy_std_mean": float("nan"), "policy_std_max": float("nan")}
            return {
                "policy_std_min": float(std.min().item()),
                "policy_std_mean": float(std.mean().item()),
                "policy_std_max": float(std.max().item()),
            }
        if getattr(policy, "noise_std_type", None) == "log" and hasattr(policy, "log_std"):
            std = torch.exp(policy.log_std.detach())
            return {
                "policy_std_min": float(std.min().item()),
                "policy_std_mean": float(std.mean().item()),
                "policy_std_max": float(std.max().item()),
            }
    return {"policy_std_min": float("nan"), "policy_std_mean": float("nan"), "policy_std_max": float("nan")}


def _is_non_finite_abort_reason(reason: str) -> bool:
    reason_lower = reason.lower()
    return "non-finite" in reason_lower or "nan" in reason_lower or "inf" in reason_lower


def iter_scalar_loss_items(loss_dict: dict[str, Any]):
    """Yield (key, float) pairs safe for TensorBoard / numeric console logging."""
    for key, value in loss_dict.items():
        if isinstance(value, bool):
            yield key, float(value)
        elif isinstance(value, (int, float)):
            numeric = float(value)
            if numeric == numeric and numeric not in (float("inf"), float("-inf")):
                yield key, numeric


def should_abort_training(
    loss_dict: dict[str, Any],
    *,
    iteration: int,
    total_batches: int,
    abort_skip_fraction: float = 0.95,
    warmup_iterations: int = 15000,
) -> tuple[bool, str]:
    """Decide whether to stop training after an update step.

    NOTE: Auto-abort is disabled. This function always returns (False, "")
    to allow training to continue regardless of numerical issues.
    Warnings are still printed to the console for monitoring.
    """
    # Auto-abort disabled - always allow training to continue
    return False, ""


def print_abort_report(
    *,
    iteration: int,
    reason: str,
    loss_dict: dict[str, Any],
    diagnostics: dict[str, Any],
    log_dir: str | None,
) -> None:
    lines = [
        "",
        "=" * 72,
        f"[TRAINING ABORT] iteration={iteration}",
        f"  reason: {reason}",
        f"  log_dir: {log_dir or '(none)'}",
        "-" * 72,
        f"  skipped_non_finite_batches: {int(loss_dict.get('skipped_non_finite_batches', 0))}"
        f"/{int(diagnostics.get('total_mini_batches', 0))}",
        f"  skipped_oversized_value_loss: {int(loss_dict.get('skipped_oversized_value_loss', 0))}",
        f"  max_value_loss_batch: {loss_dict.get('max_value_loss_batch', float('nan')):.6g}",
        f"  value_function (mean logged): {loss_dict.get('value_function', float('nan')):.6g}",
        f"  surrogate: {loss_dict.get('surrogate', float('nan')):.6g}",
        f"  amp: {loss_dict.get('amp', float('nan')):.6g}",
        f"  returns min/max/abs_max: "
        f"{loss_dict.get('returns_min', float('nan')):.4g} / "
        f"{loss_dict.get('returns_max', float('nan')):.4g} / "
        f"{loss_dict.get('returns_abs_max', float('nan')):.4g}",
        f"  policy std min/mean/max: "
        f"{diagnostics.get('policy_std_min', float('nan')):.4g} / "
        f"{diagnostics.get('policy_std_mean', float('nan')):.4g} / "
        f"{diagnostics.get('policy_std_max', float('nan')):.4g}",
        f"  learning_rate: {diagnostics.get('learning_rate', float('nan')):.6g}",
        f"  stability_warmup_iterations: {diagnostics.get('stability_warmup_iterations', 0)}",
        "-" * 72,
        "  Suggested actions:",
        f"    - Resume from an earlier checkpoint under {log_dir or 'logs/...'}",
        "    - Inspect TensorBoard: Loss/value_function, Loss/skipped_non_finite_batches",
        "    - Review motion data quality and reduce learning_rate if needed",
        "=" * 72,
        "",
    ]
    report = "\n".join(lines)
    print(report, flush=True)

    if log_dir:
        payload = {
            "iteration": iteration,
            "reason": reason,
            "loss_dict": {k: float(v) if isinstance(v, (int, float)) else v for k, v in loss_dict.items()},
            "diagnostics": diagnostics,
        }
        out_path = Path(log_dir) / "training_abort_report.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[TRAINING ABORT] Wrote {out_path}", flush=True)


def handle_training_abort(
    *,
    alg,
    loss_dict: dict[str, Any],
    iteration: int,
    log_dir: str | None = None,
) -> bool:
    """Check training stability and print warnings, but never abort.

    NOTE: Auto-abort is disabled. This function only logs warnings
    to console and TensorBoard for monitoring. Training always continues.
    """
    total_batches = total_mini_batches(alg.num_learning_epochs, alg.num_mini_batches)

    # Check for numerical issues and print warnings, but don't abort
    reason = str(loss_dict.get("training_abort_reason", ""))
    if loss_dict.get("training_abort"):
        print(f"[TRAINING WARNING] iteration={iteration}: {reason}", flush=True)

    skipped_non_finite = float(loss_dict.get("skipped_non_finite_batches", 0.0))
    if total_batches > 0 and skipped_non_finite > 0:
        print(
            f"[TRAINING WARNING] iteration={iteration}: "
            f"skipped {int(skipped_non_finite)}/{total_batches} non-finite mini-batches",
            flush=True,
        )

    # Always return False - training never aborts
    return False
