"""Run the E3 P0 smoke (MoT + MoE + Latent + overhead) without touching YOLO-Master core.

The script is executed with the deployed YOLO-Master baseline Python. It walks a
YOLO-Master checkout, runs one real forward per routing family, adapts the
``last_routing_snapshot`` modules through the existing v1 adapters into
``routing_records.jsonl``, measures hook overhead, isolates family failures and
archives evidence into a per-run ``artifacts/smoke/<run_id>/`` directory with a
SHA-256 manifest.

Example:
    python scripts/run_e3_smoke.py --config configs/e3_smoke.yaml
    python scripts/run_e3_smoke.py --config configs/e3_smoke.yaml --run-id smoke-20260905-01
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# ---------------------------------------------------------------------------
# Metrics (pure helpers, covered by unit tests)
# ---------------------------------------------------------------------------

def _to_plain(value: Any) -> Any:
    """Recursively convert tensors / numpy values to plain JSON-friendly values."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if getattr(value, "ndim", 0) == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value

def _shannon_entropy(probabilities: Any) -> float:
    """Shannon entropy in nats over a non-negative probability vector."""
    import numpy as np

    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum())

def _gini_of_loads(loads: Any) -> float:
    """Gini coefficient over a non-negative load vector (0 = perfectly even)."""
    import numpy as np

    total = float(loads.sum())
    if loads.size == 0 or total <= 0.0:
        return 0.0
    ordered = np.sort(loads)
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    numerator = 2.0 * float(np.sum(ranks * ordered))
    denominator = float(ordered.size * total)
    coefficient = numerator / denominator - (ordered.size + 1.0) / ordered.size
    return float(np.clip(coefficient, 0.0, 1.0))

def routing_metrics(expert_usage: Any) -> dict[str, Any]:
    """Summarize expert usage as normalized shares plus concentration metrics.

    Formulas follow the standard definitions: Shannon entropy H = -sum(p ln p),
    normalized entropy H / ln(E), and the Gini coefficient of the load vector.
    """
    import numpy as np

    loads = np.asarray(_to_plain(expert_usage), dtype=np.float64).reshape(-1)
    loads = np.where(np.isfinite(loads), loads, 0.0)
    loads = np.maximum(loads, 0.0)

    total_load = float(loads.sum())
    shares = loads / total_load if total_load > 0.0 else np.zeros_like(loads)

    entropy = _shannon_entropy(shares)
    denom = math.log(shares.size) if shares.size > 1 else 0.0
    normalized = entropy / denom if denom > 0.0 else 0.0

    return {
        "expert_load": shares.tolist(),
        "expert_load_sum": float(shares.sum()),
        "routing_entropy_nats": entropy,
        "routing_entropy_normalized": normalized,
        "load_gini": _gini_of_loads(loads),
        "dominant_expert_share": float(shares.max()) if shares.size else 0.0,
    }

def validate_records(records: list[dict[str, Any]]) -> list[str]:
    """Run invariant checks over captured routing records."""
    errors: list[str] = []
    for index, record in enumerate(records):
        load = record.get("expert_load")
        if not load:
            errors.append(f"record {index}: missing expert_load")
            continue
        if any(value < 0 for value in load):
            errors.append(f"record {index}: negative load value")
        if not math.isclose(sum(load), 1.0, abs_tol=1e-6):
            errors.append(f"record {index}: load sum {sum(load):.6f} != 1")
        for key in ("routing_entropy_normalized", "load_gini", "dominant_expert_share"):
            value = record.get(key)
            if value is None or not math.isfinite(value):
                errors.append(f"record {index}: non-finite {key}")
    return errors

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)

def resolve_baseline_root(config: dict[str, Any], cli_root: str | None) -> Path:
    if cli_root:
        root = Path(cli_root)
    elif os.environ.get("BASELINE_ROOT"):
        root = Path(os.environ["BASELINE_ROOT"])
    else:
        root = Path(config.get("baseline_root", "../YOLO-Master"))
        if not root.is_absolute():
            root = PACKAGE_ROOT / root
    return root.resolve()

def generate_run_id() -> str:
    """Return a fresh unique run id (timestamp + random suffix)."""
    now = datetime.now().astimezone()
    return f"smoke-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"

def resolve_run_id(config: Mapping[str, Any], cli_run_id: str | None) -> str:
    """CLI ``--run-id`` wins; otherwise every run gets a fresh unique id."""
    if cli_run_id:
        return str(cli_run_id).strip()
    return generate_run_id()

def smoke_artifacts_dir(package_root: Path, run_id: str) -> Path:
    return package_root / "artifacts" / "smoke" / run_id

def setup_logging(artifacts: Path) -> logging.Logger:
    """Configure one file + one console handler per target log path.

    Repeated calls never duplicate handlers: calling again with the same
    artifacts dir is a no-op, and calling with a different dir replaces the
    file handler so one process can run several isolated runs.
    """
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("e3-smoke")
    logger.setLevel(logging.INFO)
    target = (artifacts / "full.log").resolve()

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == target:
            return logger

    # Replace any file handler from a previous run without duplicating it.
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(artifacts / "full.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
               for handler in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

def collect_environment() -> dict[str, Any]:
    try:
        import torch
        import ultralytics
    except Exception as exc:
        raise RuntimeError(
            "ultralytics/torch not importable. Run this script with the deployed "
            f"YOLO-Master baseline Python. ({exc})"
        ) from exc
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": "<baseline-python>",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cwd": "<baseline-root>",
        "captured_at": _now_iso(),
    }

def _forward_random_image(module: Any, size: int) -> None:
    import torch

    with torch.no_grad():
        module(torch.randn(1, 3, int(size), int(size)))

def _capture_model_records(
    model_config: str,
    *,
    run_id: str,
    size: int,
    force_moe_snapshot: bool = False,
) -> list[Any]:
    """Build one real model, run one real forward, adapt snapshots to records."""
    from ultralytics import YOLO

    from scripts.routing_capture import capture_records

    model = YOLO(model_config)
    module = model.model.eval()
    if force_moe_snapshot:
        for _, mod in module.named_modules():
            if "moe" in mod.__class__.__module__.lower() and hasattr(mod, "last_routing_snapshot"):
                mod._moe_force_snapshot = True
    records = capture_records(
        module,
        run_id=run_id,
        captured_at=_now_iso(),
        forward=lambda root: _forward_random_image(root, size),
        training=False,
    )
    return records

def run_mot(config: dict[str, Any], artifacts: Path, logger: logging.Logger, *, run_id: str) -> dict[str, Any]:
    """Run the MoT diagnostic script, then capture real MoT v1 records."""
    mot = config["mot"]
    output_dir = Path(mot["output_dir"])
    if output_dir.is_dir():
        for pattern in ("*.csv", "*.png", "*.txt"):
            for file in sorted(output_dir.glob(pattern)):
                file.unlink()
    command = [sys.executable, str(Path(mot["script"])), *[str(a) for a in mot.get("args", [])]]
    logger.info("MoT: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    logger.info(result.stdout)
    if result.stderr:
        logger.info("MoT stderr: %s", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"MoT smoke failed with rc={result.returncode}")
    copied: list[str] = []
    if output_dir.is_dir():
        for pattern in ("*.csv", "*.png", "*.txt"):
            for file in sorted(output_dir.glob(pattern)):
                shutil.copy2(file, artifacts / file.name)
                copied.append(file.name)

    records = _capture_model_records(mot["model_config"], run_id=run_id, size=int(mot.get("image_size", 640)))
    if not records:
        raise RuntimeError("MoT: no routing snapshots captured after real forward")
    logger.info("MoT routing records: %d", len(records))
    return {"returncode": result.returncode, "copied_files": copied, "records": records}

def run_moe(config: dict[str, Any], artifacts: Path, logger: logging.Logger, *, run_id: str) -> dict[str, Any]:
    """Run MoE over the real validation dataset and capture v1 records."""
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker

    moe = config["moe"]
    logger.info("MoE: model=%s dataset=%s", moe["model_config"], moe.get("dataset"))
    model = YOLO(moe["model_config"])
    module = model.model
    # MoE routers only refresh their snapshot every MOE_SNAPSHOT_INTERVAL
    # forwards unless forced; request a snapshot on the val forward we run.
    for _, mod in module.named_modules():
        if "moe" in mod.__class__.__module__.lower() and hasattr(mod, "last_routing_snapshot"):
            mod._moe_force_snapshot = True
    with ExpertUsageTracker(module) as tracker:
        model.val(
            data=moe.get("dataset", "coco8.yaml"),
            split=moe.get("dataset_split", "val"),
            batch=moe.get("batch", 1),
            device=config.get("device", "cpu"),
            verbose=False,
        )
        stats = {
            layer: {str(expert_id): {"hits": expert.hits, "weighted_sum": expert.weighted_sum, "avg_weight": expert.avg_weight}
                   for expert_id, expert in experts.items()}
            for layer, experts in tracker.usage_stats.items()
        }
    hooked = getattr(tracker, "hooked_modules", None)
    if not stats:
        raise RuntimeError("MoE usage stats empty: no experts tracked")
    (artifacts / "moe_usage_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("MoE usage_stats: %s", json.dumps(stats, ensure_ascii=False, indent=2))

    # Upstream MoE routers only publish ``last_routing_snapshot`` inside a
    # training-mode forward, so run one real training-mode forward (the flags
    # above force every routed layer to refresh its snapshot this step).
    import torch

    module.train()
    _ = module(torch.randn(1, 3, 640, 640))
    del _

    from scripts.routing_capture import capture_records

    records = capture_records(module, run_id=run_id, captured_at=_now_iso(), training=True)
    if not records:
        raise RuntimeError("MoE: no last_routing_snapshot captured after val forward")
    logger.info("MoE routing records: %d", len(records))
    return {"usage_stats": stats, "hooked_modules": hooked, "records": records}

def run_latent(config: dict[str, Any], artifacts: Path, logger: logging.Logger, *, run_id: str) -> dict[str, Any]:
    """Run one real Latent forward and capture v1 records (plus legacy keys)."""
    from ultralytics import YOLO

    from scripts.routing_capture import capture_records

    latent = config["latent"]
    logger.info("Latent: model=%s size=%s", latent["model_config"], latent.get("image_size", 640))
    model = YOLO(latent["model_config"])
    module = model.model.eval()
    size = int(latent.get("image_size", 640))

    records = capture_records(
        module,
        run_id=run_id,
        captured_at=_now_iso(),
        forward=lambda root: _forward_random_image(root, size),
        training=False,
    )
    if not records:
        raise RuntimeError("no latent routing snapshots captured")
    logger.info("Latent routing records: %d", len(records))

    # Legacy keys-only summary, kept for the 8.25 admission artifact.
    keys_records: list[dict[str, Any]] = []
    for name, mod in module.named_modules():
        snapshot = getattr(mod, "last_routing_snapshot", None)
        if snapshot:
            keys = sorted(snapshot.keys())
            keys_records.append(
                {
                    "module_name": name,
                    "module_type": mod.__class__.__name__,
                    "num_keys": len(keys),
                    "keys": keys,
                }
            )
    with (artifacts / "latent_snapshot.jsonl").open("w", encoding="utf-8") as stream:
        for record in keys_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Latent legacy keys records: %d", len(keys_records))
    return {"records": records, "module_count": len(records)}

def parse_overhead_percent(text: str) -> float:
    """Parse ``overhead: 1.50%`` output, accepting optional +/- signs.

    NaN / Inf tokens never match the numeric pattern and raise ValueError, so
    non-finite overhead values are rejected instead of being recorded.
    """
    pattern = re.compile(r"overhead:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%")
    match = pattern.search(text or "")
    if match is None:
        raise ValueError("overhead_percent not found in overhead script output (missing or non-finite)")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"overhead_percent must be finite, got {match.group(1)!r}")
    return value

def run_overhead(config: dict[str, Any], artifacts: Path, logger: logging.Logger) -> dict[str, Any]:
    overhead = config["overhead"]
    command = [
        sys.executable,
        overhead["script"],
        "--model", overhead["model_config"],
        "--iterations", str(overhead.get("iterations", 50)),
        "--warmup", str(overhead.get("warmup", 5)),
        "--size", str(overhead.get("size", 640)),
    ]
    logger.info("Overhead: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    text = result.stdout + result.stderr
    logger.info(text)
    if result.returncode != 0:
        raise RuntimeError(f"overhead measurement failed with rc={result.returncode}")
    try:
        overhead_percent = parse_overhead_percent(text)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    overhead_result = {
        "returncode": result.returncode,
        "output": text,
        "overhead_percent": overhead_percent,
    }
    (artifacts / "overhead_result.json").write_text(
        json.dumps(overhead_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("overhead_percent=%s", overhead_percent)
    return {
        "returncode": result.returncode,
        "output": text,
        "overhead_percent": overhead_percent,
    }

def execute_smoke_steps(
    steps: Mapping[str, Callable[[], dict[str, Any]]],
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[Any]]:
    """Run every family step sequentially; one failure never blocks the rest.

    Returns (summary, collected_records).  Summary carries started_at /
    finished_at / status / error plus one entry per step with its own status,
    error, and timing.
    """
    started_at = _now_iso()
    summary: dict[str, Any] = {"status": "PASS", "steps": {}, "started_at": started_at}
    collected: list[Any] = []
    failures: list[tuple[str, str]] = []
    for name, step_fn in steps.items():
        step_started = _now_iso()
        try:
            result = step_fn()
        except Exception as exc:  # noqa: BLE001
            logger.error("%s step failed: %s", name, exc)
            summary["steps"][name] = {
                "status": "FAIL",
                "error": str(exc),
                "started_at": step_started,
                "finished_at": _now_iso(),
            }
            summary["status"] = "FAIL"
            failures.append((name, str(exc)))
            continue
        step: dict[str, Any] = {"status": "PASS", "started_at": step_started, "finished_at": _now_iso()}
        records = result.get("records") or []
        if records:
            step["records"] = len(records)
            collected.extend(records)
        summary["steps"][name] = step
    summary["finished_at"] = _now_iso()
    if failures:
        summary["error"] = "; ".join(f"{name}: {message}" for name, message in failures)
    return summary, collected

def build_manifest(artifacts: Path) -> dict[str, str]:
    return {
        file.name: sha256_file(file)
        for file in sorted(artifacts.iterdir())
        if file.is_file() and file.name != "manifest.sha256.json"
    }

def load_mot_records(csv_path: Path) -> list[dict[str, Any]]:
    """Read ``mot_routing_detailed.csv`` into per-decision validation records."""
    if not csv_path.is_file():
        raise RuntimeError(f"MoT detailed CSV missing: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"MoT detailed CSV is empty: {csv_path}")
    decisions: dict[tuple[str, str, str], list[float]] = {}
    try:
        for row in rows:
            key = (row["scene"], row["image_id"], row["layer"])
            decisions.setdefault(key, []).append(float(row["top1_share"]))
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"cannot parse MoT detailed CSV {csv_path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for key, loads in sorted(decisions.items()):
        scene, image_id, layer = key
        records.append({"scene": scene, "image_id": image_id, "layer": layer, **routing_metrics(loads)})
    return records

def verify_artifacts(artifacts: Path) -> list[str]:
    """Return post-condition errors for the MoE/Latent/overhead evidence files."""
    errors: list[str] = []

    moe_json = artifacts / "moe_usage_stats.json"
    try:
        moe_stats = json.loads(moe_json.read_text(encoding="utf-8")) if moe_json.is_file() else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        moe_stats = None
        errors.append(f"MoE usage stats unreadable: {exc}")
    if moe_stats is None:
        errors.append("MoE usage stats missing or empty")
    elif not isinstance(moe_stats, dict):
        raise RuntimeError(f"MoE usage stats must be a JSON object, got {type(moe_stats).__name__}")
    elif not moe_stats:
        errors.append("MoE usage stats missing or empty")

    latent_jsonl = artifacts / "latent_snapshot.jsonl"
    valid_records = 0
    if latent_jsonl.is_file():
        try:
            lines = latent_jsonl.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"latent_snapshot.jsonl unreadable: {exc}")
            lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                errors.append("latent_snapshot.jsonl contains an invalid record")
                break
            valid_records += 1
    if valid_records == 0:
        errors.append("latent_snapshot.jsonl missing or has no valid records")

    overhead_json = artifacts / "overhead_result.json"
    try:
        overhead = json.loads(overhead_json.read_text(encoding="utf-8")) if overhead_json.is_file() else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        overhead = None
        errors.append(f"overhead_result.json unreadable: {exc}")
    if overhead is None:
        errors.append("overhead_percent missing or null")
    elif not isinstance(overhead, dict):
        raise RuntimeError(f"overhead_result.json must be a JSON object, got {type(overhead).__name__}")
    elif overhead.get("overhead_percent") is None:
        errors.append("overhead_percent missing or null")

    return errors

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the E3 P0 smoke package")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "e3_smoke.yaml"))
    parser.add_argument("--baseline-root", default=None, help="Override the deployed YOLO-Master checkout path")
    parser.add_argument("--run-id", default=None, help="Explicit run id; defaults to a fresh unique id per run")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    baseline_root = resolve_baseline_root(config, args.baseline_root)
    if not baseline_root.is_dir():
        raise SystemExit(f"baseline_root not found: {baseline_root}")
    os.chdir(baseline_root)

    run_id = resolve_run_id(config, args.run_id)
    artifacts = smoke_artifacts_dir(PACKAGE_ROOT, run_id)
    artifacts.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(artifacts)
    logger.info("E3 P0 smoke starting. run_id=%s baseline_root=%s", run_id, baseline_root)

    environment = collect_environment()
    with (artifacts / "environment.json").open("w", encoding="utf-8") as stream:
        json.dump(environment, stream, ensure_ascii=False, indent=2)

    summary: dict[str, Any] = {
        "scope": "E3 P0 smoke; MoT/MoE/Latent unified v1 capture + overhead; isolated steps",
        "run_id": run_id,
        "schema_version": config.get("schema_version"),
        "official_base_ref": config.get("official_base_ref"),
        "baseline_root": str(baseline_root),
        "environment": environment,
    }

    steps: dict[str, Callable[[], dict[str, Any]]] = {
        "mot": lambda: run_mot(config, artifacts, logger, run_id=run_id),
        "moe": lambda: run_moe(config, artifacts, logger, run_id=run_id),
        "latent": lambda: run_latent(config, artifacts, logger, run_id=run_id),
        "overhead": lambda: run_overhead(config, artifacts, logger),
    }
    step_summary, records = execute_smoke_steps(steps, logger)
    summary.update(step_summary)

    routing_path = artifacts / "routing_records.jsonl"
    if records:
        from scripts.routing_capture import write_records

        written = write_records(records, routing_path)
        logger.info("routing_records.jsonl written: %d records -> %s", written, routing_path)
    else:
        logger.warning("no routing records collected; routing_records.jsonl not written")

    if summary["status"] == "PASS":
        failures: list[str] = []
        try:
            failures.extend(verify_artifacts(artifacts))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"artifact validation failed: {exc}")
        mot_records: list[dict[str, Any]] = []
        try:
            mot_records = load_mot_records(artifacts / "mot_routing_detailed.csv")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"MoT CSV load failed: {exc}")
        validation_errors = validate_records(mot_records) if mot_records else ["no MoT records to validate"]
        summary["validation"] = {
            "errors": validation_errors,
            "mot_records": len(mot_records),
            "mot": "see CSV artifacts",
        }
        if validation_errors:
            failures.append("MoT validation failed: " + "; ".join(validation_errors[:5]))
        if failures:
            logger.error("Smoke validation failed: %s", "; ".join(failures))
            summary["status"] = "FAIL"
            summary["error"] = "; ".join(failures)

    with (artifacts / "config.resolved.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)

    with (artifacts / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    with (artifacts / "manifest.sha256.json").open("w", encoding="utf-8") as stream:
        json.dump(build_manifest(artifacts), stream, ensure_ascii=False, indent=2)

    logger.info("result=%s", summary["status"])
    print(f"result={summary['status']}")
    return 0 if summary["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
