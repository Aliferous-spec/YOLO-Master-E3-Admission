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
    python scripts/run_e3_smoke.py --verify-artifacts artifacts/smoke/smoke-20260905-01
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
                try:
                    file.unlink()
                except OSError as exc:
                    logger.warning("MoT stale-output cleanup skipped %s: %s", file, exc)
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


# ---------------------------------------------------------------------------
# P1-A sample-level capture (per-sample snapshots -> sample_routing_records.jsonl)
# ---------------------------------------------------------------------------


def _sample_records_path(artifacts: Path) -> Path:
    """Return the dedicated P1-A sample JSONL path inside an artifacts dir."""
    return Path(artifacts) / "sample_routing_records.jsonl"


def _set_moe_snapshot_force(model: Any) -> int:
    """Force every routed MoE module to refresh its snapshot each forward."""
    forced = 0
    for _, mod in model.named_modules():
        if "moe" in mod.__class__.__module__.lower() and hasattr(mod, "last_routing_snapshot"):
            mod._moe_force_snapshot = True
            forced += 1
    return forced


def _load_baseline_module(baseline_root: Path, script_rel: str) -> Any:
    """Import a YOLO-Master script module by file path (avoids package clashes).

    The deployed baseline ships its own ``scripts`` package, which would shadow
    this package, so the file is loaded under a unique module name.
    """
    import importlib.util

    script_path = (baseline_root / script_rel).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"baseline script not found: {script_path}")
    module_name = f"_e3_p1_baseline_{script_path.stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load baseline script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _mot_sample_tensors(config: Mapping[str, Any], baseline_root: Path, size: int) -> list[Any]:
    """Return MoT sample tensors from the upstream synthetic scene generator.

    Upstream order (fixed): sparse_small / dense_small / large_regular /
    irregular_occluded, matching the P0 CSV diagnostic input set.
    """
    mot = config["mot"]
    module = _load_baseline_module(baseline_root, str(mot["script"]))
    scenes = module.synthetic_scenes(int(size), 1)
    return [tensor for _name, tensor, _image_ids in scenes]


def _moe_val_sample_tensors(yolo: Any, config: Mapping[str, Any]) -> list[Any]:
    """Return the coco8 val image tensors exactly as the YOLO val pipeline feeds them.

    Reuses the upstream detection validator dataset + preprocessing (letterbox,
    rect batch shape, /255) so the samples are the same inputs the P0 MoE
    ``ExpertUsageTracker`` evidence sees; external-id association is ordinal.
    """
    import torch
    from ultralytics.data.utils import check_det_dataset

    moe = config["moe"]
    device = str(config.get("device", "cpu"))
    args = {
        **yolo.overrides,
        "rect": True,
        "data": moe.get("dataset", "coco8.yaml"),
        "split": moe.get("dataset_split", "val"),
        "batch": int(moe.get("batch", 1)),
        "device": device,
        "verbose": False,
        "mode": "val",
    }
    validator = yolo._smart_load("validator")(args=args, _callbacks={})
    validator.training = False
    validator.args.workers = 0
    validator.args.rect = True
    validator.data = check_det_dataset(validator.args.data, split=validator.args.split)
    validator.stride = int(yolo.model.stride.max().item())
    validator.device = torch.device(device)
    loader = validator.get_dataloader(validator.data.get(validator.args.split), validator.args.batch)
    return [validator.preprocess(batch)["img"] for batch in loader]


def _capture_mot_sample_records(config: Mapping[str, Any], baseline_root: Path, run_id: str) -> list[Any]:
    """Run one eval forward per MoT synthetic scene; return module-level rows."""
    from ultralytics import YOLO

    from scripts.routing_capture import capture_sample_records

    mot = config["mot"]
    size = int(mot.get("image_size", 640))
    tensors = _mot_sample_tensors(config, baseline_root, size)
    if len(tensors) != 4:
        raise RuntimeError(f"MoT sample source must yield 4 synthetic scenes, got {len(tensors)}")
    model = YOLO(mot["model_config"]).model.eval()
    records = capture_sample_records(
        model, tensors, run_id=run_id, captured_at=_now_iso(), training=False
    )
    if not records:
        raise RuntimeError("MoT sample capture produced no routing records")
    return records


def _capture_moe_sample_records(config: Mapping[str, Any], run_id: str) -> list[Any]:
    """Run one train-mode forward per coco8 val image with BN state isolation."""
    from ultralytics import YOLO

    from scripts.routing_capture import (
        capture_sample_records,
        restore_bn_running_state,
        snapshot_bn_running_state,
    )

    moe = config["moe"]
    yolo = YOLO(moe["model_config"])
    model = yolo.model.train()
    if _set_moe_snapshot_force(model) == 0:
        raise RuntimeError("MoE sample capture: no routed MoE modules to force snapshots")
    tensors = _moe_val_sample_tensors(yolo, config)
    if len(tensors) != 4:
        raise RuntimeError(f"MoE sample source must yield 4 coco8 val images, got {len(tensors)}")
    bn_snapshot = snapshot_bn_running_state(model)
    if not bn_snapshot:
        raise RuntimeError("MoE sample capture: no track_running_stats BatchNorm found to isolate")

    def _restore_bn(root: Any, step: int) -> None:
        restore_bn_running_state(root, bn_snapshot)

    records = capture_sample_records(
        model, tensors, run_id=run_id, captured_at=_now_iso(), training=True, before_each=_restore_bn
    )
    if not records:
        raise RuntimeError("MoE sample capture produced no routing records")
    return records


def _capture_latent_sample_records(config: Mapping[str, Any], run_id: str) -> list[Any]:
    """Run one eval forward on the single random latent input; return rows."""
    import torch
    from ultralytics import YOLO

    from scripts.routing_capture import capture_sample_records

    latent = config["latent"]
    size = int(latent.get("image_size", 640))
    model = YOLO(latent["model_config"]).model.eval()
    records = capture_sample_records(
        model, [torch.randn(1, 3, size, size)], run_id=run_id, captured_at=_now_iso(), training=False
    )
    if not records:
        raise RuntimeError("Latent sample capture produced no routing records")
    return records


def run_sample_capture(config: Mapping[str, Any], artifacts: Path, logger: logging.Logger, *, run_id: str, baseline_root: Path) -> dict[str, Any]:
    """P1-A step: per-sample snapshots into ``sample_routing_records.jsonl``.

    The step writes its own dedicated JSONL (``write_records`` on the sample
    path) and returns no ``records``, so ``execute_smoke_steps()`` never merges
    sample rows into the canonical ``routing_records.jsonl``.
    """
    from scripts.routing_capture import write_records

    captures = (
        ("mot", lambda: _capture_mot_sample_records(config, baseline_root, run_id)),
        ("moe", lambda: _capture_moe_sample_records(config, run_id)),
        ("latent", lambda: _capture_latent_sample_records(config, run_id)),
    )
    family_rows: list[Any] = []
    details: dict[str, Any] = {}
    for family, capture in captures:
        records = capture()
        family_rows.extend(records)
        steps = sorted({record.step for record in records})
        modules = {record.module.name for record in records}
        details[family] = {"samples": len(steps), "modules": len(modules), "records": len(records)}
        logger.info(
            "P1-A sample capture %s: samples=%d modules=%d records=%d",
            family,
            len(steps),
            len(modules),
            len(records),
        )
    if not family_rows:
        raise RuntimeError("P1-A sample capture produced no rows at all")
    sample_path = _sample_records_path(artifacts)
    written = write_records(family_rows, sample_path)
    logger.info("P1-A sample rows written: %d -> %s", written, sample_path)
    return {"sample_file": sample_path.name, "sample_records": written, "families": details}


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


def verify_manifest(artifacts: Path) -> list[str]:
    """Verify an artifacts directory: manifest presence, SHA-256, run_id.

    Returns a list of human-readable errors; an empty list means the run is
    consistent. Missing or tampered files always fail with their path.
    """
    artifacts = Path(artifacts)
    errors: list[str] = []
    manifest_path = artifacts / "manifest.sha256.json"
    if not manifest_path.is_file():
        return [f"manifest missing: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"manifest unreadable: {manifest_path}: {exc}"]
    if not isinstance(manifest, dict):
        return [f"manifest must be a JSON object mapping file name to sha256: {manifest_path}"]

    for name, expected in sorted(manifest.items()):
        path = artifacts / name
        if not path.is_file():
            errors.append(f"file missing: {path}")
            continue
        actual = sha256_file(path)
        if actual != str(expected):
            errors.append(f"sha256 mismatch: {path} (expected {expected}, got {actual})")

    run_id = artifacts.name
    summary_path = artifacts / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"summary unreadable: {summary_path}: {exc}")
        summary = None
    if summary is None:
        errors.append(f"summary missing: {summary_path}")
    elif summary.get("run_id") != run_id:
        errors.append(
            f"run_id mismatch: summary run_id {summary.get('run_id')!r} != artifacts dir name {run_id!r}"
        )
    return errors


def verify_sample_records(sample_path: Path, *, run_id: str) -> list[str]:
    """P1-A sample-file validation: all rows v1, run_id-consistent, continuous.

    Returns a list of human-readable errors (empty list means valid).  Every
    sample row must parse as ``e3-routing/v1`` with ``step`` an int >= 0; per
    family the steps must be 0..S_f-1 with the same module set on every step,
    and the ``(run_id, family, module.name, step)`` primary key must be unique.
    """
    from scripts.routing_record_writer import iter_records

    errors: list[str] = []
    sample_path = Path(sample_path)
    if not sample_path.is_file():
        return [f"sample routing records missing: {sample_path}"]
    try:
        rows = list(iter_records(sample_path))
    except ValueError as exc:
        return [f"sample routing records invalid: {exc}"]
    if not rows:
        return [f"sample routing records empty: {sample_path}"]

    per_family: dict[str, dict[int, set[str]]] = {}
    seen_keys: set[tuple[str, str, str, int]] = set()
    for row in rows:
        if row.run_id != run_id:
            errors.append(f"sample row run_id mismatch: {row.run_id!r} != {run_id!r}")
        if not isinstance(row.step, int) or row.step < 0:
            errors.append(
                f"sample row must have an int step >= 0: module={row.module.name!r} step={row.step!r}"
            )
            continue
        key = (row.run_id, row.family, row.module.name, row.step)
        if key in seen_keys:
            errors.append(f"duplicate sample row key: {key}")
        seen_keys.add(key)
        per_family.setdefault(row.family, {}).setdefault(row.step, set()).add(row.module.name)

    for family, step_modules in sorted(per_family.items()):
        sample_count = max(step_modules) + 1
        if set(step_modules) != set(range(sample_count)):
            errors.append(f"family {family}: sample steps not continuous: {sorted(step_modules)}")
            continue
        module_sets = [step_modules[step] for step in range(sample_count)]
        base = module_sets[0]
        if any(modules != base for modules in module_sets[1:]):
            errors.append(f"family {family}: routed module set differs across samples")
            continue
        actual = sum(len(modules) for modules in module_sets)
        expected = sample_count * len(base)
        if actual != expected:
            errors.append(
                f"family {family}: row count {actual} != samples {sample_count} x modules {len(base)}"
            )
    return errors


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


def verify_cli_artifacts(artifacts: Path) -> int:
    errors = verify_manifest(artifacts)
    sample_path = Path(artifacts) / "sample_routing_records.jsonl"
    if sample_path.is_file():
        errors.extend(verify_sample_records(sample_path, run_id=artifacts.name))
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"result=FAIL ({len(errors)} error(s))")
        return 1
    print(f"result=PASS manifest OK: {artifacts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the E3 P0 smoke package")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "e3_smoke.yaml"))
    parser.add_argument("--baseline-root", default=None, help="Override the deployed YOLO-Master checkout path")
    parser.add_argument("--run-id", default=None, help="Explicit run id; defaults to a fresh unique id per run")
    parser.add_argument(
        "--verify-artifacts", default=None, metavar="DIR",
        help="Verify an artifacts directory (manifest + SHA-256 + run_id) and exit",
    )
    args = parser.parse_args(argv)

    if args.verify_artifacts:
        return verify_cli_artifacts(Path(args.verify_artifacts))

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
        "samples": lambda: run_sample_capture(
            config, artifacts, logger, run_id=run_id, baseline_root=baseline_root
        ),
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
        sample_path = artifacts / "sample_routing_records.jsonl"
        if sample_path.is_file():
            failures.extend(verify_sample_records(sample_path, run_id=run_id))
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

    # Log the final result before hashing, then close the file handler so the
    # manifest hashes full.log in its final, flushed state.
    logger.info("result=%s", summary["status"])
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

    with (artifacts / "manifest.sha256.json").open("w", encoding="utf-8") as stream:
        json.dump(build_manifest(artifacts), stream, ensure_ascii=False, indent=2)

    print(f"result={summary['status']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
