"""Run the E3 smoke across several seeds, then verify every run (P0 red line).

The task book requires at least 3 seeds for P0 evidence.  This driver runs
``scripts.run_e3_smoke`` once per seed (a per-seed temp config overrides the
``seed`` field; the on-disk config is never modified), then verifies each run
with ``--verify-artifacts`` and prints one summary table.

Exit code 0 only when every seed's smoke AND manifest verification pass.

Usage (acceptance env):
    python -m scripts.run_smoke_seeds --baseline-root D:/YOLO-Master
    python -m scripts.run_smoke_seeds --baseline-root D:/YOLO-Master --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _run_one(seed: int, config_path: Path, baseline_root: Path, run_id: str) -> dict[str, Any]:
    """Run one smoke with the given seed via a temp config; return a summary row."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["seed"] = seed
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", prefix=f"e3-smoke-seed{seed}-", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        temp_config = Path(handle.name)

    command = [
        sys.executable, "-m", "scripts.run_e3_smoke",
        "--config", str(temp_config),
        "--baseline-root", str(baseline_root),
        "--run-id", run_id,
    ]
    # Strip PYTHONPATH/PYTHONHOME: ambient shims (e.g. safe-delete wrappers)
    # have previously hijacked os.remove inside the smoke process and broken
    # the MoT step. The smoke must run with only its own venv on sys.path.
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME")
    }
    env["PYTHONUTF8"] = "1"
    try:
        proc = subprocess.run(command, cwd=PACKAGE_ROOT, env=env, capture_output=True, text=True)
        exit_code = proc.returncode
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    finally:
        temp_config.unlink(missing_ok=True)

    run_dir = PACKAGE_ROOT / "artifacts" / "smoke" / run_id
    steps = {}
    if (run_dir / "summary.json").is_file():
        try:
            steps = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            steps = {}

    verify = subprocess.run(
        [sys.executable, "-m", "scripts.run_e3_smoke", "--verify-artifacts", str(run_dir)],
        cwd=PACKAGE_ROOT, env=env, capture_output=True, text=True,
    )
    verify_ok = verify.returncode == 0
    verify_tail = (verify.stdout + verify.stderr).strip().splitlines()[-1] if verify.stdout else ""

    return {
        "seed": seed,
        "run_id": run_id,
        "exit_code": exit_code,
        "smoke_pass": exit_code == 0,
        "verify_pass": verify_ok,
        "verify_tail": verify_tail,
        "log_tail": tail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the E3 smoke across seeds and verify every run")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "e3_smoke.yaml"))
    parser.add_argument("--baseline-root", default=None, help="Deployed YOLO-Master checkout path")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seed list (default 0,1,2)")
    args = parser.parse_args(argv)

    baseline_root = Path(
        args.baseline_root
        or os.environ.get("BASELINE_ROOT")
        or yaml.safe_load(Path(args.config).read_text(encoding="utf-8")).get("baseline_root", "../YOLO-Master")
    ).resolve()
    if not baseline_root.is_dir():
        print(f"baseline_root not found: {baseline_root} — pass --baseline-root D:/path/to/YOLO-Master")
        return 2

    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    for seed in seeds:
        run_id = f"seed{seed}-{stamp}-{abs(hash((seed, stamp))) % 0xFFFF:04x}"
        print(f"[seed {seed}] run_id={run_id} ...", flush=True)
        rows.append(_run_one(seed, Path(args.config), baseline_root, run_id))

    print()
    print(f"{'seed':>4}  {'smoke':>5}  {'verify':>6}  run_id")
    all_ok = True
    for row in rows:
        smoke = "PASS" if row["smoke_pass"] else f"FAIL({row['exit_code']})"
        verify = "PASS" if row["verify_pass"] else "FAIL"
        print(f"{row['seed']:>4}  {smoke:>5}  {verify:>6}  {row['run_id']}")
        if not row["verify_pass"]:
            print(f"      verify: {row['verify_tail']}")
        all_ok = all_ok and row["smoke_pass"] and row["verify_pass"]
    print(f"\nresult={'PASS' if all_ok else 'FAIL'} ({len(rows)} seed run(s), baseline={baseline_root})")
    if not all_ok:
        for row in rows:
            if not row["smoke_pass"]:
                for line in row["log_tail"]:
                    print(f"[seed {row['seed']}] {line}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
