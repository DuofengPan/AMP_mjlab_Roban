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


def should_abort_training(
    loss_dict: dict[str, Any],
    *,
    total_batches: int,
    abort_skip_fraction: float = 0.95,
) -> tuple[bool, str]:
    if loss_dict.get("training_abort"):
        return True, str(loss_dict.get("training_abort_reason", "numerical instability"))

    skipped = float(loss_dict.get("skipped_non_finite_batches", 0.0))
    if total_batches > 0 and skipped >= total_batches * abort_skip_fraction:
        return True, f"skipped {int(skipped)}/{total_batches} mini-batches (>={100 * abort_skip_fraction:.0f}%)"

    max_value_loss = float(loss_dict.get("max_value_loss_batch", 0.0))
    max_value_limit = float(loss_dict.get("stability_max_value_loss", 0.0))
    if max_value_limit > 0.0 and max_value_loss > max_value_limit:
        return True, f"value_loss spike (max={max_value_loss:.4g} > limit={max_value_limit:.4g})"

    returns_abs_max = float(loss_dict.get("returns_abs_max", 0.0))
    max_return_limit = float(loss_dict.get("stability_max_abs_return", 0.0))
    if max_return_limit > 0.0 and returns_abs_max > max_return_limit:
        return True, f"returns outlier (abs_max={returns_abs_max:.4g} > limit={max_return_limit:.4g})"

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
    log_dir: str | None,
    abort_skip_fraction: float,
) -> bool:
    """Return True if training should stop after this iteration."""
    total_batches = total_mini_batches(alg.num_learning_epochs, alg.num_mini_batches)
    diagnostics = {
        "total_mini_batches": total_batches,
        "learning_rate": float(getattr(alg, "learning_rate", float("nan"))),
        **policy_std_snapshot(alg.policy),
    }
    should_abort, reason = should_abort_training(
        loss_dict,
        total_batches=total_batches,
        abort_skip_fraction=abort_skip_fraction,
    )
    if not should_abort:
        return False

    print_abort_report(
        iteration=iteration,
        reason=reason,
        loss_dict=loss_dict,
        diagnostics=diagnostics,
        log_dir=log_dir,
    )
    return True
