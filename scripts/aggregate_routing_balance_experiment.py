r"""Aggregate the routing balance intervention runs (3 seeds x 2 arms).

Reads one ``intervention_result.json`` per run from a runs root, verifies the
run set is complete (no missing seed/arm, every measured epoch and target layer
present, every coefficient assertion green), then compares the arms with seeds
as the paired unit.

Pre-registered criterion (fixed before the runs; never adjusted afterwards):
    pass  <=>  mean paired layer-Gini delta < 0
               and paired bootstrap 95% CI upper bound < 0
               and normalized entropy moves the consistent way
               (mean delta > 0 and CI lower bound > 0)

A null or negative result is reported as measured; this module never tunes the
coefficient, seeds, epochs or thresholds to reach a conclusion.

Example:
    python -m scripts.aggregate_routing_balance_experiment --runs-root ^
        artifacts/routing_intervention/<experiment-id>
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


SCHEMA_VERSION = "e3-routing-balance-experiment/v1"
GENERATOR = "scripts/aggregate_routing_balance_experiment.py"
ARMS = ("baseline", "intervention")
METRICS = ("gini", "entropy_normalized", "top1_share")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CRITERION = (
    "intervention mean layer-Gini < baseline AND paired bootstrap 95% CI upper < 0; "
    "normalized entropy consistent (mean delta > 0 AND CI lower > 0)"
)
RESULT_NAME = "intervention_result.json"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Pure helpers (covered by unit tests)
# ---------------------------------------------------------------------------


def load_runs(runs_root: Path) -> list[dict[str, Any]]:
    """Load every ``intervention_result.json`` directly under ``runs_root``."""
    paths = sorted(Path(runs_root).glob(f"*/{RESULT_NAME}"))
    if not paths:
        raise FileNotFoundError(f"no */{RESULT_NAME} under {runs_root}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def verify_runs(runs: Sequence[Mapping[str, Any]], *, seeds: Sequence[int]) -> list[str]:
    """Return integrity problems; empty means the run set is complete and clean."""
    problems: list[str] = []
    expected = {(arm, int(seed)) for arm in ARMS for seed in seeds}
    seen: dict[tuple[str, int], Mapping[str, Any]] = {}
    for run in runs:
        key = (str(run.get("arm")), int(run.get("parameters", {}).get("seed", -1)))
        if key in seen:
            problems.append(f"duplicate run for {key}")
        seen[key] = run
    for key in sorted(expected - set(seen)):
        problems.append(f"missing run for arm={key[0]} seed={key[1]}")
    for key in sorted(set(seen) - expected):
        problems.append(f"unexpected run for arm={key[0]} seed={key[1]}")
    for key, run in sorted(seen.items()):
        problems.extend(_verify_one(key, run))
    return problems


def _verify_one(key: tuple[str, int], run: Mapping[str, Any]) -> list[str]:
    label = f"{key[0]}/seed{key[1]}"
    issues: list[str] = []
    params = run.get("parameters", {})
    rows = run.get("layer_metrics") or []
    if not rows:
        return [f"{label}: no layer_metrics"]
    measured = int(params.get("measured_epochs", 0))
    epochs = {int(row["epoch"]) for row in rows}
    if len(epochs) != measured:
        issues.append(f"{label}: measured epochs {sorted(epochs)} != {measured}")
    modules = {str(row["module"]) for row in rows}
    if len(modules) != len(run.get("target_modules") or {}):
        issues.append(f"{label}: layers {sorted(modules)} do not match the target modules")
    drifted = [a for a in run.get("assertions", []) if not a.get("ok")]
    if drifted:
        issues.append(f"{label}: {len(drifted)} coefficient assertion(s) failed")
    if int(run.get("nonfinite_count", 0)) > 0:
        issues.append(f"{label}: {run['nonfinite_count']} non-finite observation(s)")
    return issues


def metric_mean(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    """Mean of ``metric`` over rows; raises on an empty selection."""
    if not rows:
        raise ValueError(f"no rows to average {metric}")
    return float(statistics.fmean(float(row[metric]) for row in rows))


def seed_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """``{arm: {seed: {metric means, epochs, wall_seconds}}}`` over measured rows."""
    summary: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for run in runs:
        arm = str(run["arm"])
        seed = int(run["parameters"]["seed"])
        rows = run["layer_metrics"]
        entry: dict[str, Any] = {metric: metric_mean(rows, metric) for metric in METRICS}
        entry["epochs"] = int(run["parameters"]["measured_epochs"])
        entry["metric_rows"] = len(rows)
        entry["run_id"] = run.get("run_id")
        entry["wall_seconds"] = float(run.get("wall_seconds", 0.0))
        entry["gini_by_layer"] = {
            str(module): metric_mean([r for r in rows if r["module"] == module], "gini")
            for module in sorted({str(r["module"]) for r in rows})
        }
        summary[arm][seed] = entry
    return summary


def paired_deltas(summary: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, dict[int, float]]:
    """Per-seed ``intervention - baseline`` for each metric."""
    seeds = sorted(summary["baseline"])
    if seeds != sorted(summary["intervention"]):
        raise ValueError("baseline and intervention cover different seeds")
    deltas: dict[str, dict[int, float]] = {metric: {} for metric in METRICS}
    for seed in seeds:
        for metric in METRICS:
            deltas[metric][seed] = float(summary["intervention"][seed][metric]) - float(
                summary["baseline"][seed][metric]
            )
    return deltas


def evaluate(
    deltas: Mapping[str, Mapping[int, float]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Per-metric paired statistics plus the frozen pass criterion."""
    from scripts.measure_sample_capture_overhead import _bootstrap_ci_mean, _distribution

    stats: dict[str, Any] = {}
    for metric in METRICS:
        values = [float(deltas[metric][key]) for key in sorted(deltas[metric])]
        entry = _distribution(values)
        entry["ci95"] = _bootstrap_ci_mean(values, resamples=resamples, seed=seed)
        entry["per_seed"] = {str(key): float(deltas[metric][key]) for key in sorted(deltas[metric])}
        stats[metric] = entry
    gini = stats["gini"]
    entropy = stats["entropy_normalized"]
    gini_pass = bool(gini["mean"] < 0.0 and gini["ci95"][1] < 0.0)
    entropy_pass = bool(entropy["mean"] > 0.0 and entropy["ci95"][0] > 0.0)
    return {
        "metrics": stats,
        "gini_criterion_met": gini_pass,
        "entropy_criterion_met": entropy_pass,
        "pass": bool(gini_pass and entropy_pass),
        "criterion": CRITERION,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_epoch_csv(path: Path, runs: Sequence[Mapping[str, Any]]) -> int:
    """One row per arm/seed/epoch/layer."""
    fields = [
        "arm", "seed", "epoch", "module", "module_type", "num_experts", "top_k",
        "gini", "entropy_nats", "entropy_normalized", "top1_share", "dominant_expert",
        "aux_loss", "aux_loss_finite", "mixture_aux_loss", "epoch_seconds",
    ]
    written = 0
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for row in run["layer_metrics"]:
                writer.writerow({name: row.get(name) for name in fields})
                written += 1
    return written


def write_seed_csv(path: Path, summary: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> int:
    """One row per arm/seed."""
    fields = ["arm", "seed", "epochs", "metric_rows", *METRICS, "wall_seconds", "run_id"]
    written = 0
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in ARMS:
            for seed in sorted(summary[arm]):
                entry = summary[arm][seed]
                row = {name: entry.get(name) for name in fields}
                row["arm"] = arm
                row["seed"] = seed
                writer.writerow(row)
                written += 1
    return written


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Short, factual report of the measured comparison."""
    summary = payload["seed_summary"]
    verdict = payload["comparison"]
    lines = [
        "# Routing balance intervention: baseline vs moe_loss_fn.balance_loss_coeff=4.0",
        "",
        f"- schema: `{payload['schema_version']}`",
        f"- generated: {payload['generated_at']}",
        f"- runs root: `{payload['runs_root']}`",
        "- arms: baseline = `balance_loss_coeff=1.0`, intervention = `moe_loss_fn.balance_loss_coeff=4.0`",
        f"- seeds: {payload['seeds']}; measured epochs per run: {payload['measured_epochs']}",
        f"- criterion (pre-registered): {verdict['criterion']}",
        "",
        "## Per-seed means",
        "",
        "| arm | seed | mean Gini | mean normalized entropy | mean top-1 share | wall s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        for seed in sorted(summary[arm]):
            entry = summary[arm][seed]
            lines.append(
                f"| {arm} | {seed} | {entry['gini']:.6f} | {entry['entropy_normalized']:.6f} | "
                f"{entry['top1_share']:.6f} | {entry['wall_seconds']:.1f} |"
            )
    lines += [
        "",
        "## Paired comparison (intervention - baseline, seed as the paired unit)",
        "",
        "| metric | mean delta | 95% CI | per-seed delta |",
        "| --- | --- | --- | --- |",
    ]
    for metric in METRICS:
        entry = verdict["metrics"][metric]
        per_seed = ", ".join(f"seed{k}={v:+.6f}" for k, v in entry["per_seed"].items())
        lines.append(
            f"| {metric} | {entry['mean']:+.6f} | "
            f"[{entry['ci95'][0]:+.6f}, {entry['ci95'][1]:+.6f}] | {per_seed} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- Gini criterion met: **{verdict['gini_criterion_met']}**",
        f"- entropy consistency met: **{verdict['entropy_criterion_met']}**",
        f"- overall: **{'PASS' if verdict['pass'] else 'NOT PASS'}**",
        "",
        "Scope: routing-structure observables on a 4-image coco8 workload with a short "
        "CPU schedule (2 warmup + 10 measured epochs). This is not an accuracy, convergence "
        "or real-scale training claim. A null result is reported as measured.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="Directory holding the per-run output dirs")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated expected seeds")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow rewriting result files in --runs-root")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = [int(part) for part in str(args.seeds).split(",") if part.strip()]
    runs_root = Path(args.runs_root).resolve()
    try:
        runs = load_runs(runs_root)
    except (OSError, ValueError) as exc:
        print(f"aggregate_routing_balance_experiment: {exc}", file=sys.stderr)
        return 2
    problems = verify_runs(runs, seeds=seeds)
    if problems:
        print("aggregate_routing_balance_experiment: incomplete run set:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    summary = seed_summary(runs)
    comparison = evaluate(paired_deltas(summary))
    measured_epochs = int(runs[0]["parameters"]["measured_epochs"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "generated_at": _now_iso(),
        "runs_root": str(runs_root),
        "seeds": seeds,
        "measured_epochs": measured_epochs,
        "integrity": {
            "runs_found": len(runs),
            "expected_runs": len(ARMS) * len(seeds),
            "problems": problems,
        },
        "seed_summary": summary,
        "comparison": comparison,
        "environment": runs[0].get("environment"),
        "baselines": {arm: run.get("parameters") for arm, run in
                      ((str(r["arm"]), r) for r in runs)},
    }

    epoch_csv = runs_root / "epoch_metrics.csv"
    seed_csv = runs_root / "seed_summary.csv"
    comparison_json = runs_root / "comparison.json"
    report_md = runs_root / "routing_balance_result.md"
    for path in (epoch_csv, seed_csv, comparison_json, report_md):
        if path.exists() and not args.overwrite:
            print(f"refusing to overwrite existing {path} (pass --overwrite)", file=sys.stderr)
            return 2
    rows = write_epoch_csv(epoch_csv, runs)
    seeds_written = write_seed_csv(seed_csv, summary)
    comparison_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_md.write_text(render_markdown(payload), encoding="utf-8")

    stats = comparison["metrics"]
    print(f"runs: {len(runs)} ({seeds_written} arm/seed rows, {rows} epoch metric rows)")
    for metric in METRICS:
        entry = stats[metric]
        print(
            f"  {metric}: mean delta {entry['mean']:+.6f} "
            f"95% CI [{entry['ci95'][0]:+.6f}, {entry['ci95'][1]:+.6f}]"
        )
    print(f"verdict: {'PASS' if comparison['pass'] else 'NOT PASS'} ({comparison['criterion']})")
    for path in (epoch_csv, seed_csv, comparison_json, report_md):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
