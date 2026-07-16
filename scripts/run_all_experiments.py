#!/usr/bin/env python3
"""
Master Experiment Runner for HWIN-Net Phase XIX Scientific Validation

Orchestrates the complete experimental campaign:
1. BENCH-001..BENCH-005: Main benchmarks
2. ABLA-001..ABLA-007: Ablation studies
3. THEO-001..THEO-007: Theory validation
4. ROBU-001..ROBU-006: Robustness tests
5. CALI-001..CALI-007: Calibration analysis
6. FAIL-001..FAIL-005: Failure analysis
7. DISC-001..DISC-006: Scientific discovery
8. EFF-001: Efficiency measurement
9. STAT-001..STAT-005: Statistical analysis

Usage:
    python scripts/run_all_experiments.py --phase bench
    python scripts/run_all_experiments.py --phase ablation
    python scripts/run_all_experiments.py --phase all
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_registry import get_registry, ExperimentCategory


def run_experiment_script(script_name, args=None):
    """Run an experiment script module."""
    cmd = [sys.executable, "-m", "scripts.experiments." + script_name]
    if args:
        cmd.extend(args)
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def check_data_exists(data_dir):
    """Check if HWIN-Bench data exists."""
    data_path = Path(data_dir)
    required = ["train.parquet", "val.parquet", "test.parquet"]
    for f in required:
        if not (data_path / f).exists():
            print("[ERROR] Missing data file: " + str(data_path / f))
            print("[ERROR] Please place HWIN-Bench data in " + str(data_dir))
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run HWIN-Net experiments")
    parser.add_argument("--phase", type=str, choices=[
        "bench", "ablation", "theory", "robustness", "calibration", 
        "failure", "discovery", "efficiency", "statistical", "all"
    ], default="all", help="Which phase to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Specific seeds to run")
    parser.add_argument("--experiments", type=str, nargs="+", default=None, help="Specific experiment IDs")
    args = parser.parse_args()

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print("=" * 60)
    print("HWIN-Net Phase XIX: Scientific Validation Campaign")
    print("=" * 60)
    
    # Check data
    from utils.config import load_config
    config = load_config("configs/config.yaml")
    data_dir = config.get("data", {}).get("train_data_path", "./data")
    data_dir = str(Path(data_dir).parent)
    
    if not check_data_exists(data_dir):
        print("[ERROR] HWIN-Bench data not found!")
        print("Please place train.parquet, val.parquet, test.parquet in " + data_dir)
        return 1

    phase_map = {
        "bench": ("run_benchmark", ["--experiments", "BENCH-001", "BENCH-002", "BENCH-003", "BENCH-004", "BENCH-005"]),
        "ablation": ("run_ablation", []),
        "theory": ("run_theory", []),
        "robustness": ("run_robustness", []),
        "calibration": ("run_calibration", []),
        "failure": ("run_failure", []),
        "discovery": ("run_discovery", []),
        "efficiency": ("run_efficiency", []),
        "statistical": ("run_statistical", []),
    }

    if args.phase == "all":
        phases = list(phase_map.keys())
    else:
        phases = [args.phase]

    for phase in phases:
        if phase not in phase_map:
            print("[WARN] Unknown phase: " + phase)
            continue
        
        script, default_args = phase_map[phase]
        cmd_args = args.experiments if args.experiments else default_args
        if args.seeds:
            cmd_args = cmd_args + ["--seeds"] + [str(s) for s in args.seeds]
        
        print("\n" + "=" * 60)
        print("PHASE: " + phase.upper())
        print("=" * 60)
        
        success = run_experiment_script(script, cmd_args)
        if not success:
            print("[ERROR] Phase " + phase + " failed!")
            return 1

    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
