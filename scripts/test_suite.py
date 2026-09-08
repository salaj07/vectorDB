"""
VectorForge Master Test Suite & Validation Script

Runs unit tests, integration tests, API tests, and benchmark verification.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DEFAULT_N, DEFAULT_D


def run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n========================================================")
    print(f" STEP: {name}")
    print(f" COMMAND: {' '.join(cmd)}")
    print(f"========================================================")
    result = subprocess.run(cmd, text=True)
    if result.returncode == 0:
        print(f"SUCCESS: {name}")
        return True
    else:
        print(f"FAILED: {name} (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="VectorForge Master Test Suite")
    parser.add_argument(
        "-n",
        "--num-vectors",
        type=int,
        default=DEFAULT_N,
        help="Number of vectors for benchmark (default: 50,000)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (1,000 vectors)"
    )
    args = parser.parse_args()

    n = 1000 if args.quick else args.num_vectors

    steps = [
        ("Unit Tests", [sys.executable, "-m", "pytest", "tests/unit", "-v"]),
        ("Integration Tests", [sys.executable, "-m", "pytest", "tests/integration", "-v"]),
        ("API Tests", [sys.executable, "-m", "pytest", "tests/api", "-v"]),
        (
            "Generate Synthetic Dataset",
            [
                sys.executable,
                "scripts/generate_dataset.py",
                "-n",
                str(n),
                "-d",
                str(DEFAULT_D),
            ],
        ),
        ("Build Indexes", [sys.executable, "scripts/build_indexes.py"]),
        ("Generate Ground Truth", [sys.executable, "scripts/generate_ground_truth.py"]),
        ("Run Benchmarks", [sys.executable, "scripts/benchmark.py"]),
    ]

    failed = []
    for name, cmd in steps:
        success = run_step(name, cmd)
        if not success:
            failed.append(name)

    print("\n========================================================")
    print(" SUMMARY")
    print("========================================================")
    if not failed:
        print("ALL STEPS PASSED SUCCESSFULLY! VectorForge is fully verified.")
        sys.exit(0)
    else:
        print(f"FAILED STEPS: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
