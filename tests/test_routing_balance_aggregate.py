"""Tests for the routing balance experiment aggregator.

``scripts/aggregate_routing_balance_experiment.py`` only reads artifacts, so its
pure surface is tested here with synthetic runs: a complete set must verify, an
incomplete or drifted set must be rejected, and the pass criterion must follow
the frozen rule instead of the data.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_routing_balance_experiment import (
    CRITERION,
    evaluate,
    metric_mean,
    paired_deltas,
    seed_summary,
    verify_runs,
    write_seed_csv,
)

MODULES = ("model.5", "model.8", "model.11")


def _run(
    arm: str,
    seed: int,
    gini: float,
    entropy: float,
    *,
    top1: float = 0.5,
    measured: int = 2,
    assertions_ok: bool = True,
    nonfinite: int = 0,
) -> dict:
    rows = [
        {
            "arm": arm,
            "seed": seed,
            "epoch": epoch,
            "module": module,
            "gini": gini,
            "entropy_normalized": entropy,
            "top1_share": top1,
        }
        for epoch in range(measured)
        for module in MODULES
    ]
    return {
        "schema_version": "e3-routing-intervention/v1",
        "run_id": f"{arm}-seed{seed}",
        "arm": arm,
        "wall_seconds": 10.0,
        "nonfinite_count": nonfinite,
        "parameters": {"seed": seed, "measured_epochs": measured},
        "target_modules": {module: "DetailAwareLowRankHybridAdaptiveGateMoE" for module in MODULES},
        "assertions": [{"when": "on_train_end", "epoch": None, "ok": assertions_ok}],
        "layer_metrics": rows,
    }


def _complete_set(gini_baseline: float, gini_intervention: float,
                  entropy_baseline: float, entropy_intervention: float) -> list[dict]:
    runs = []
    for seed in (0, 1, 2):
        runs.append(_run("baseline", seed, gini_baseline, entropy_baseline))
        runs.append(_run("intervention", seed, gini_intervention, entropy_intervention))
    return runs


def test_verify_accepts_a_complete_set() -> None:
    assert verify_runs(_complete_set(0.8, 0.5, 0.3, 0.4), seeds=[0, 1, 2]) == []


def test_verify_reports_a_missing_seed() -> None:
    runs = [r for r in _complete_set(0.8, 0.5, 0.3, 0.4)
            if not (r["arm"] == "intervention" and r["parameters"]["seed"] == 1)]
    problems = verify_runs(runs, seeds=[0, 1, 2])
    assert any("missing run for arm=intervention seed=1" in p for p in problems)


def test_verify_rejects_failed_assertions_and_nonfinite() -> None:
    runs = _complete_set(0.8, 0.5, 0.3, 0.4)
    runs[0] = _run("baseline", 0, 0.8, 0.3, assertions_ok=False, nonfinite=2)
    problems = verify_runs(runs, seeds=[0, 1, 2])
    assert any("assertion" in p for p in problems)
    assert any("non-finite" in p for p in problems)


def test_metric_mean_and_seed_summary() -> None:
    runs = _complete_set(0.8, 0.5, 0.30, 0.40)
    summary = seed_summary(runs)
    assert summary["baseline"][0]["gini"] == pytest.approx(0.8)
    assert summary["intervention"][2]["entropy_normalized"] == pytest.approx(0.4)
    assert set(summary["baseline"][0]["gini_by_layer"]) == set(MODULES)
    assert metric_mean([r for r in runs[0]["layer_metrics"]], "gini") == pytest.approx(0.8)


def test_paired_deltas_subtract_baseline_from_intervention() -> None:
    deltas = paired_deltas(seed_summary(_complete_set(0.8, 0.5, 0.30, 0.40)))
    assert deltas["gini"] == {0: pytest.approx(-0.3), 1: pytest.approx(-0.3), 2: pytest.approx(-0.3)}
    assert deltas["entropy_normalized"][0] == pytest.approx(0.1)


def test_evaluate_passes_when_gini_drops_and_entropy_rises() -> None:
    verdict = evaluate(paired_deltas(seed_summary(_complete_set(0.8, 0.5, 0.30, 0.40))))
    assert verdict["metrics"]["gini"]["ci95"][1] < 0.0
    assert verdict["metrics"]["entropy_normalized"]["ci95"][0] > 0.0
    assert verdict["pass"] is True
    assert verdict["criterion"] == CRITERION


def test_evaluate_does_not_pass_on_a_null_result() -> None:
    verdict = evaluate(paired_deltas(seed_summary(_complete_set(0.8, 0.8, 0.30, 0.30))))
    assert verdict["metrics"]["gini"]["mean"] == pytest.approx(0.0)
    assert verdict["pass"] is False


def test_evaluate_does_not_pass_when_gini_falls_but_entropy_disagrees() -> None:
    verdict = evaluate(paired_deltas(seed_summary(_complete_set(0.8, 0.5, 0.40, 0.30))))
    assert verdict["gini_criterion_met"] is True
    assert verdict["entropy_criterion_met"] is False
    assert verdict["pass"] is False


def test_seed_csv_carries_the_arm_and_seed_columns(tmp_path: Path) -> None:
    summary = seed_summary(_complete_set(0.8, 0.5, 0.30, 0.40))
    path = tmp_path / "seed_summary.csv"

    assert write_seed_csv(path, summary) == 6

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert {(row["arm"], row["seed"]) for row in rows} == {
        (arm, str(seed)) for arm in ("baseline", "intervention") for seed in (0, 1, 2)
    }
