r"""Routing balance intervention: one real training session per arm.

Single confirmed lever: ``moe_loss_fn.balance_loss_coeff`` on routed MoE blocks.
Setting ``module.balance_loss_coeff`` alone does NOT change the training
objective: the MoE auxiliary loss reads the ``MoELoss`` instance attribute once
per forward. This script therefore never touches the module attribute, and the
isolated test in ``tests/test_routing_balance_intervention.py`` pins that
contract.

The intervention is applied at ``on_train_start`` and re-checked on every
``on_train_epoch_start`` and at ``on_train_end``, because an upstream mixture
controller could in principle overwrite the attribute afterwards. A session
that cannot hold the coefficient fails loudly (non-zero exit) instead of
silently producing a mixed-arm result.

Per measured epoch (``epoch >= --warmup-epochs``) and per routed MoE layer this
script records Gini, normalized entropy, top-1 share, dominant expert, the
model-level mixture auxiliary loss and the epoch wall-clock, using the existing
unmodified ``scripts.routing_capture`` chain. Aggregation across arms/seeds
lives in ``scripts/aggregate_routing_balance_experiment.py``.

Boundaries
    - No YOLO-Master file is modified; the knob is a plain Python attribute.
    - No canonical evidence is touched: outputs land in the run's own
      ``--out-dir``.
    - The baseline checkout is located through the environment
      (``--baseline-root`` / ``BASELINE_ROOT`` / ``../YOLO-Master``); no machine
      path is hard-coded.

Environment prerequisite: the deployed venv needs ``POLARS_SKIP_CPU_CHECK=1``
(see ``_preflight_polars``), otherwise the trainer aborts every session with a
misleading non-finite-state error.

Example (one session):
    set PYTHONPATH=D:\YOLO-Master
    set POLARS_SKIP_CPU_CHECK=1
    python -m scripts.run_routing_balance_intervention --arm intervention ^
        --seed 0 --epochs 12 --warmup-epochs 2 --run-id <id> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

SCHEMA_VERSION = "e3-routing-intervention/v1"
GENERATOR = "scripts/run_routing_balance_intervention.py"
DEFAULT_MODEL = "ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml"
DEFAULT_DATA = "ultralytics/cfg/datasets/coco8.yaml"
DEFAULT_BASELINE_ROOT = os.environ.get("BASELINE_ROOT", "../YOLO-Master")
DEFAULT_INTERVENTION_COEFF = 4.0
ARMS = ("baseline", "intervention")
LOSS_FN_ATTR = "moe_loss_fn"
KNOB_ATTR = "balance_loss_coeff"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve_cfg_path(baseline_root: Path, value: str) -> Path:
    """Absolute path for a model/data cfg: absolute or relative to baseline_root."""
    path = Path(value)
    return path if path.is_absolute() else (baseline_root / path).resolve()


# ---------------------------------------------------------------------------
# Pure knob helpers (no torch; covered by unit tests)
# ---------------------------------------------------------------------------


def _iter_targets(model: Any) -> Iterator[tuple[str, Any, Any]]:
    """Yield (module path, module, loss_fn) for every module that owns the knob."""
    for name, module in model.named_modules():
        loss_fn = getattr(module, LOSS_FN_ATTR, None)
        if loss_fn is not None and hasattr(loss_fn, KNOB_ATTR):
            yield name, module, loss_fn


def tune_balance_coeff(model: Any, coeff: float) -> list[dict[str, Any]]:
    """Set ONLY ``moe_loss_fn.balance_loss_coeff`` on every target module.

    ``module.balance_loss_coeff`` is deliberately left alone because the
    auxiliary loss never reads it; mutating it would misreport what changed.
    Raises when no target exists so an empty intervention cannot pass silently.
    """
    coeff = float(coeff)
    rows: list[dict[str, Any]] = []
    for name, module, loss_fn in _iter_targets(model):
        before = float(getattr(loss_fn, KNOB_ATTR))
        setattr(loss_fn, KNOB_ATTR, coeff)
        module_attr = getattr(module, KNOB_ATTR, None)
        rows.append(
            {
                "module": name,
                "module_type": type(module).__name__,
                "knob": f"{LOSS_FN_ATTR}.{KNOB_ATTR}",
                "before": before,
                "after": float(getattr(loss_fn, KNOB_ATTR)),
                "module_attr_untouched": None if module_attr is None else float(module_attr),
            }
        )
    if not rows:
        raise RuntimeError(f"no module exposes {LOSS_FN_ATTR}.{KNOB_ATTR}")
    return rows


def read_balance_coeffs(model: Any) -> dict[str, float]:
    """Snapshot ``{module path: moe_loss_fn.balance_loss_coeff}`` for all targets."""
    return {name: float(getattr(loss_fn, KNOB_ATTR)) for name, _, loss_fn in _iter_targets(model)}


def assert_balance_coeff(model: Any, coeff: float) -> dict[str, Any]:
    """Raise unless every target module still holds ``coeff``; return the evidence."""
    coeff = float(coeff)
    observed = read_balance_coeffs(model)
    if not observed:
        raise RuntimeError(f"no module exposes {LOSS_FN_ATTR}.{KNOB_ATTR}")
    drifted = {name: value for name, value in observed.items() if value != coeff}
    if drifted:
        raise RuntimeError(f"balance coefficient drifted off {coeff}: {drifted}")
    return {"expected": coeff, "observed": observed, "ok": True}


def metric_row(
    record: Any, *, arm: str, seed: int, epoch: int, epoch_seconds: float | None,
    mixture_aux_loss: float | None,
) -> dict[str, Any]:
    """Flatten one v1 routing record into a per-epoch metric row."""
    routing = record.routing
    return {
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "module": record.module.name,
        "module_type": record.module.type,
        "num_experts": int(routing.num_experts),
        "top_k": int(routing.top_k),
        "gini": float(routing.load_gini),
        "entropy_nats": float(routing.routing_entropy_nats),
        "entropy_normalized": float(routing.routing_entropy_normalized),
        "top1_share": float(routing.dominant_expert_share),
        "dominant_expert": int(routing.dominant_expert),
        "aux_loss": None if record.aux_loss.value is None else float(record.aux_loss.value),
        "aux_loss_finite": record.aux_loss.finite,
        "mixture_aux_loss": mixture_aux_loss,
        "epoch_seconds": epoch_seconds,
    }


def _force_moe_snapshot(root: Any) -> int:
    """Force per-forward MoE snapshot refresh (the P1-A observation convention)."""
    forced = 0
    for _, module in root.named_modules():
        if "moe" in module.__class__.__module__.lower() and hasattr(module, "last_routing_snapshot"):
            module._moe_force_snapshot = True
            forced += 1
    return forced


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def out_dir_of(cfg: dict[str, Any]) -> Path:
    """Artifact directory for one session."""
    return Path(cfg["out_dir"])


def _train_session(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run one real YOLO training session and collect per-epoch routing metrics."""
    import torch

    from ultralytics import YOLO

    from scripts.routing_capture import capture_records
    from scripts.routing_record_writer import RoutingRecordWriter

    seed = int(cfg["seed"])
    arm = cfg["arm"]
    coeff = float(cfg["coeff"])
    warmup_epochs = int(cfg["warmup_epochs"])
    expected = coeff if arm == "intervention" else None

    torch.manual_seed(seed)
    model = YOLO(str(cfg["model"]))

    state: dict[str, Any] = {
        "module_types": {},
        "baseline_coeffs": {},
        "tuned": [],
        "assertions": [],
        "forced_snapshot_modules": 0,
        "records_per_epoch": {},
        "records_total": 0,
        "epoch_seconds": [],
        "layer_metrics": [],
        "nonfinite": [],
    }

    def _baseline_target() -> float:
        values = set(state["baseline_coeffs"].values())
        if len(values) != 1:
            raise RuntimeError(f"baseline arm has mixed coefficients: {state['baseline_coeffs']}")
        return float(values.pop())

    def _record_assertion(when: str, trainer: Any, epoch: int | None = None) -> None:
        target = expected if expected is not None else _baseline_target()
        evidence = assert_balance_coeff(trainer.model, target)
        state["assertions"].append({"when": when, "epoch": epoch, **evidence})

    def _on_train_start(trainer: Any) -> None:
        state["forced_snapshot_modules"] = _force_moe_snapshot(trainer.model)
        state["baseline_coeffs"] = read_balance_coeffs(trainer.model)
        state["module_types"] = {
            name: type(module).__name__ for name, module, _ in _iter_targets(trainer.model)
        }
        if arm == "intervention":
            state["tuned"] = tune_balance_coeff(trainer.model, coeff)
        _record_assertion("on_train_start", trainer)

    def _on_train_epoch_start(trainer: Any) -> None:
        state["epoch_started_at"] = time.perf_counter()
        state["forced_snapshot_modules"] = _force_moe_snapshot(trainer.model)
        _record_assertion("on_train_epoch_start", trainer, int(trainer.epoch))

    def _on_train_epoch_end(trainer: Any) -> None:
        epoch = int(trainer.epoch)
        started = state.pop("epoch_started_at", None)
        seconds = None if started is None else round(time.perf_counter() - started, 4)
        records = capture_records(
            trainer.model,
            run_id=cfg["run_id"],
            captured_at=_now_iso(),
            step=epoch,
            training=True,
        )
        for record in records:
            state["writer"].write(record)
        state["records_per_epoch"][str(epoch)] = len(records)
        state["records_total"] += len(records)
        measured = epoch >= warmup_epochs
        state["epoch_seconds"].append({"epoch": epoch, "seconds": seconds, "measured": measured})
        if not measured:
            return
        aux = getattr(trainer.model, "_last_mixture_aux_loss", None)
        aux_value = None
        if aux is not None:
            try:
                aux_value = float(aux)
            except (TypeError, ValueError):
                aux_value = None
        if aux_value is not None and not math.isfinite(aux_value):
            state["nonfinite"].append({"epoch": epoch, "field": "_last_mixture_aux_loss"})
        for record in records:
            if record.aux_loss.finite is False:
                state["nonfinite"].append(
                    {"epoch": epoch, "module": record.module.name, "field": "aux_loss"}
                )
            state["layer_metrics"].append(
                metric_row(
                    record,
                    arm=arm,
                    seed=seed,
                    epoch=epoch,
                    epoch_seconds=seconds,
                    mixture_aux_loss=aux_value,
                )
            )

    def _on_train_end(trainer: Any) -> None:
        _record_assertion("on_train_end", trainer)

    state["writer"] = RoutingRecordWriter(
        out_dir_of(cfg) / f"seed{seed}_{arm}_records.jsonl", append=True
    )
    model.add_callback("on_train_start", _on_train_start)
    model.add_callback("on_train_epoch_start", _on_train_epoch_start)
    model.add_callback("on_train_epoch_end", _on_train_epoch_end)
    model.add_callback("on_train_end", _on_train_end)
    try:
        model.train(
            data=str(cfg["data"]),
            epochs=int(cfg["epochs"]),
            imgsz=int(cfg["imgsz"]),
            batch=int(cfg["batch"]),
            device=cfg["device"],
            workers=int(cfg["workers"]),
            seed=seed,
            deterministic=True,
            patience=int(cfg["epochs"]) + 1,
            project=str(out_dir_of(cfg) / "runs"),
            name=f"seed{seed}_{arm}",
            exist_ok=True,
            save=False,
            plots=False,
            verbose=False,
            cache=False,
        )
    finally:
        state["writer"].close()
    state.pop("writer", None)
    state.pop("epoch_started_at", None)
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline-root",
        default=DEFAULT_BASELINE_ROOT,
        help="Deployed YOLO-Master checkout (default BASELINE_ROOT or ../YOLO-Master)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model cfg, absolute or relative to --baseline-root"
    )
    parser.add_argument(
        "--data", default=DEFAULT_DATA, help="Dataset cfg, absolute or relative to --baseline-root"
    )
    parser.add_argument(
        "--arm",
        choices=ARMS,
        default="intervention",
        help="baseline = leave the knob at its default; intervention = set it",
    )
    parser.add_argument(
        "--coeff",
        type=float,
        default=DEFAULT_INTERVENTION_COEFF,
        help=f"balance coefficient for --arm intervention (default {DEFAULT_INTERVENTION_COEFF})",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1, help="Total epochs including warmup")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Unmeasured prefix epochs; measured epochs are --epochs minus this",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--run-id", default=None, help="Artifact run_id (default timestamp-based)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Artifact dir (default artifacts/routing_intervention/<run_id>)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing an existing --out-dir")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration only; never trains")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    baseline_root = Path(args.baseline_root).resolve()
    if not baseline_root.is_dir():
        raise FileNotFoundError(f"baseline root not found: {baseline_root}")
    model = _resolve_cfg_path(baseline_root, args.model)
    data = _resolve_cfg_path(baseline_root, args.data)
    for label, path in (("model cfg", model), ("data cfg", data)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if int(args.epochs) < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if not 0 <= int(args.warmup_epochs) < int(args.epochs):
        raise ValueError(
            f"--warmup-epochs must satisfy 0 <= warmup < epochs, got "
            f"{args.warmup_epochs} / {args.epochs}"
        )
    run_id = args.run_id or f"routing-balance-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (PACKAGE_ROOT / "artifacts" / "routing_intervention" / run_id)
    )
    return {
        "baseline_root": baseline_root,
        "model": model,
        "data": data,
        "arm": args.arm,
        "coeff": float(args.coeff),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "warmup_epochs": int(args.warmup_epochs),
        "measured_epochs": int(args.epochs) - int(args.warmup_epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": str(args.device),
        "workers": int(args.workers),
        "run_id": run_id,
        "out_dir": out_dir.resolve(),
        "output": out_dir.resolve() / "intervention_result.json",
    }


def _print_dry_run(cfg: dict[str, Any]) -> None:
    print(f"baseline_root : {cfg['baseline_root']}")
    print(f"model cfg     : {cfg['model']}  [exists]")
    print(f"data cfg      : {cfg['data']}  [exists]")
    print(f"arm           : {cfg['arm']}")
    print(f"coeff         : {cfg['coeff'] if cfg['arm'] == 'intervention' else 'unchanged (baseline)'}")
    print(f"seed/epochs   : {cfg['seed']} / {cfg['epochs']} ({cfg['warmup_epochs']} warmup + "
          f"{cfg['measured_epochs']} measured)")
    print(
        f"imgsz/batch   : {cfg['imgsz']} / {cfg['batch']} ({cfg['device']}, workers {cfg['workers']})"
    )
    print(f"knob          : {LOSS_FN_ATTR}.{KNOB_ATTR} (module attribute is never touched)")
    print(f"out dir       : {cfg['out_dir']}")
    print("dry-run       : OK (validation passed; no model was built, no training was run)")


def _environment() -> dict[str, Any]:
    import platform

    import torch
    import ultralytics

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "ultralytics_file": str(ultralytics.__file__),
        "polars_skip_cpu_check": os.environ.get("POLARS_SKIP_CPU_CHECK"),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }


def _after_coeff(state: dict[str, Any], module: str) -> Any:
    for row in state["tuned"]:
        if row["module"] == module:
            return row["after"]
    return state["baseline_coeffs"].get(module)


def _preflight_polars() -> None:
    """Surface a polars CPU-check failure before training starts.

    The trainer imports polars while verifying its healthy recovery checkpoint;
    when that import raises, the trainer reports a misleading non-finite-state
    error instead. Calling this first turns it into an actionable message.
    """
    try:
        import polars  # noqa: F401
    except RuntimeError as exc:
        raise RuntimeError(
            f"polars import failed ({exc}); the trainer needs it for healthy-checkpoint "
            "verification. Set POLARS_SKIP_CPU_CHECK=1 to bypass the CPU feature check."
        ) from exc


def _run(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    _preflight_polars()
    environment = _environment()
    output = Path(cfg["output"])
    if output.exists() and not args.overwrite:
        print(f"refusing to overwrite existing {output} (pass --overwrite)", file=sys.stderr)
        return 2
    out_dir_of(cfg).mkdir(parents=True, exist_ok=True)
    started_at = _now_iso()
    wall_started = time.perf_counter()
    print(
        f"[{cfg['arm']} seed {cfg['seed']}] training {cfg['epochs']} epoch(s) "
        f"({cfg['warmup_epochs']} warmup + {cfg['measured_epochs']} measured) ...",
        flush=True,
    )
    state = _train_session(cfg)
    wall_seconds = round(time.perf_counter() - wall_started, 4)
    finished_at = _now_iso()

    resolved_file = str(environment.get("ultralytics_file", ""))
    if str(cfg["baseline_root"]) not in resolved_file:
        print(
            f"WARNING: ultralytics resolved to {resolved_file}, not under {cfg['baseline_root']}",
            file=sys.stderr,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "run_id": cfg["run_id"],
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": wall_seconds,
        "scope": (
            "one real training session of the routing balance intervention. Routing metrics are "
            "structural observations, not an accuracy or convergence claim."
        ),
        "arm": cfg["arm"],
        "coefficient": cfg["coeff"] if cfg["arm"] == "intervention" else None,
        "parameters": {
            "baseline_root": str(cfg["baseline_root"]),
            "model_config": str(cfg["model"]),
            "data": str(cfg["data"]),
            "seed": cfg["seed"],
            "epochs": cfg["epochs"],
            "warmup_epochs": cfg["warmup_epochs"],
            "measured_epochs": cfg["measured_epochs"],
            "imgsz": cfg["imgsz"],
            "batch": cfg["batch"],
            "device": cfg["device"],
            "workers": cfg["workers"],
        },
        "environment": environment,
        "knob": {
            "path": f"{LOSS_FN_ATTR}.{KNOB_ATTR}",
            "module_attr_left_untouched": KNOB_ATTR,
            "forced_snapshot_modules": state["forced_snapshot_modules"],
        },
        "target_modules": state["module_types"],
        "baseline_coeffs": state["baseline_coeffs"],
        "tuned": state["tuned"],
        "assertions": state["assertions"],
        "records_total": state["records_total"],
        "records_per_epoch": state["records_per_epoch"],
        "epoch_seconds": state["epoch_seconds"],
        "nonfinite": state["nonfinite"],
        "nonfinite_count": len(state["nonfinite"]),
        "layer_metrics": state["layer_metrics"],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    applied = bool(state["tuned"])
    print("target modules:")
    for module, module_type in state["module_types"].items():
        print(
            f"  {module} ({module_type}): {state['baseline_coeffs'].get(module)} -> "
            f"{_after_coeff(state, module)}  [{'applied' if applied else 'baseline'}]"
        )
    print("assertions:")
    for item in state["assertions"]:
        print(f"  {item['when']} epoch={item['epoch']}: ok={item['ok']} observed={item['observed']}")
    print(
        f"records: {state['records_total']} routing record(s); measured metric rows: "
        f"{len(state['layer_metrics'])}; nonfinite: {len(state['nonfinite'])}"
    )
    print(f"wall: {wall_seconds:.1f}s")
    print(f"wrote {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = resolve_config(args)
    except (OSError, ValueError) as exc:
        print(f"run_routing_balance_intervention: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        _print_dry_run(cfg)
        return 0
    try:
        return _run(cfg, args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"run_routing_balance_intervention: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
