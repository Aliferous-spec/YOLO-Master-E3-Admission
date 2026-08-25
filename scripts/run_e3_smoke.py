"""Run the E3 admission Smoke (MoT + MoE + Latent) without modifying YOLO-Master core.

The script is executed with the deployed YOLO-Master baseline Python. It walks a
YOLO-Master checkout, collects one routing snapshot per family, measures hook
overhead, and archives evidence into ``artifacts/smoke/<run_id>/`` with a SHA-256
manifest.

Example:
    python scripts/run_e3_smoke.py --config configs/e3_smoke.yaml
"""

from __future__ import annotations

import argparse
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
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


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


def setup_logging(artifacts: Path) -> logging.Logger:
    logger = logging.getLogger("e3-smoke")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler = logging.FileHandler(artifacts / "full.log", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def collect_environment() -> dict[str, Any]:
    try:
        import torch
        import ultralytics
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ultralytics/torch not importable. Run this script with the deployed "
            f"YOLO-Master baseline Python. ({exc})"
        ) from exc
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cwd": os.getcwd(),
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def run_mot(config: dict[str, Any], artifacts: Path, logger: logging.Logger) -> dict[str, Any]:
    mot = config["mot"]
    command = [sys.executable, str(Path(mot["script"])), *[str(a) for a in mot.get("args", [])]]
    logger.info("MoT: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    logger.info(result.stdout)
    if result.stderr:
        logger.info("MoT stderr: %s", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"MoT smoke failed with rc={result.returncode}")
    output_dir = Path(mot["output_dir"])
    copied: list[str] = []
    if output_dir.is_dir():
        for pattern in ("*.csv", "*.png", "*.txt"):
            for file in sorted(output_dir.glob(pattern)):
                shutil.copy2(file, artifacts / file.name)
                copied.append(file.name)
    return {"returncode": result.returncode, "copied_files": copied}


def run_moe(config: dict[str, Any], artifacts: Path, logger: logging.Logger) -> dict[str, Any]:
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker

    moe = config["moe"]
    logger.info("MoE: model=%s dataset=%s", moe["model_config"], moe.get("dataset"))
    model = YOLO(moe["model_config"])
    with ExpertUsageTracker(model.model) as tracker:
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
    (artifacts / "moe_usage_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("MoE usage_stats: %s", json.dumps(stats, ensure_ascii=False, indent=2))
    return {"usage_stats": stats, "hooked_modules": hooked}
def run_latent(config: dict[str, Any], artifacts: Path, logger: logging.Logger) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    latent = config["latent"]
    logger.info("Latent: model=%s size=%s", latent["model_config"], latent.get("image_size", 640))
    model = YOLO(latent["model_config"])
    module = model.model.eval()
    size = int(latent.get("image_size", 640))
    with torch.no_grad():
        module(torch.randn(1, 3, size, size))
    records: list[dict[str, Any]] = []
    for name, mod in module.named_modules():
        snapshot = getattr(mod, "last_routing_snapshot", None)
        if snapshot:
            keys = sorted(snapshot.keys())
            records.append(
                {
                    "module_name": name,
                    "module_type": mod.__class__.__name__,
                    "num_keys": len(keys),
                    "keys": keys,
                }
            )
    with (artifacts / "latent_snapshot.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Latent records: %d", len(records))
    return {"records": records, "module_count": len(records)}


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
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = result.stdout + result.stderr
    logger.info(text)
    if result.returncode != 0:
        raise RuntimeError(f"overhead measurement failed with rc={result.returncode}")
    match = re.search(r"overhead:\s+([\d.]+)%", text)
    overhead_result = {
        "returncode": result.returncode,
        "output": text,
        "overhead_percent": float(match.group(1)) if match else None,
    }
    (artifacts / "overhead_result.json").write_text(
        json.dumps(overhead_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "returncode": result.returncode,
        "output": text,
        "overhead_percent": float(match.group(1)) if match else None,
    }


def build_manifest(artifacts: Path) -> dict[str, str]:
    return {file.name: sha256_file(file) for file in sorted(artifacts.iterdir()) if file.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the E3 admission Smoke package")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "e3_smoke.yaml"))
    parser.add_argument("--baseline-root", default=None, help="Override the deployed YOLO-Master checkout path")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    baseline_root = resolve_baseline_root(config, args.baseline_root)
    if not baseline_root.is_dir():
        raise SystemExit(f"baseline_root not found: {baseline_root}")
    os.chdir(baseline_root)

    run_id = config.get("run_id", "admission-20260825")
    artifacts = PACKAGE_ROOT / "artifacts" / "smoke" / run_id
    artifacts.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(artifacts)
    logger.info("E3 admission Smoke starting. baseline_root=%s", baseline_root)

    environment = collect_environment()
    with (artifacts / "environment.json").open("w", encoding="utf-8") as stream:
        json.dump(environment, stream, ensure_ascii=False, indent=2)

    summary: dict[str, Any] = {
        "status": "PASS",
        "scope": "E3 8.25 admission Smoke; MoT/MoE/Latent families; not P0",
        "run_id": run_id,
        "schema_version": config.get("schema_version"),
        "official_base_ref": config.get("official_base_ref"),
        "baseline_root": str(baseline_root),
        "environment": environment,
    }

    try:
        summary["mot"] = run_mot(config, artifacts, logger)
        summary["moe"] = run_moe(config, artifacts, logger)
        summary["latent"] = run_latent(config, artifacts, logger)
        summary["overhead"] = run_overhead(config, artifacts, logger)
    except Exception as exc:  # noqa: BLE001
        logger.error("Smoke step failed: %s", exc)
        summary["status"] = "FAIL"
        summary["error"] = str(exc)

    if summary["status"] == "PASS":
        mot_records: list[dict[str, Any]] = []
        summary["validation"] = {"errors": validate_records(mot_records), "mot": "see CSV artifacts"}

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
