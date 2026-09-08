"""
run_pipeline.py — Master orchestrator.
Executes all test suites, dataset generation, and benchmarks sequentially.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

PIPELINE_STEPS = [
    {
        "name": "1. Exact Index Unit Tests",
        "cmd": [PYTHON, str(ROOT_DIR / "tests" / "test_exact.py")],
    },
    {
        "name": "2. IVF-Flat Baseline Tests",
        "cmd": [PYTHON, str(ROOT_DIR / "tests" / "test_ivf.py")],
    },
    {
        "name": "3. Advanced IVF Tests (K-Means++, SQ8, Dynamic Routing)",
        "cmd": [PYTHON, str(ROOT_DIR / "tests" / "test_ivf_advanced.py")],
    },
    {
        "name": "4. Ground Truth Generation (50,000 Synthetic Vectors)",
        "cmd": [PYTHON, str(ROOT_DIR / "src" / "evaluate.py")],
    },
    {
        "name": "5. Synthetic Dataset Benchmark (QPS & Pareto)",
        "cmd": [PYTHON, str(ROOT_DIR / "src" / "benchmark.py"), "--dataset", "synthetic"],
    },
    {
        "name": "6. Text Corpus Embedding (sentence-transformers)",
        "cmd": [PYTHON, str(ROOT_DIR / "src" / "embed_text.py")],
    },
    {
        "name": "7. Real-Text Dataset Benchmark (QPS & Pareto)",
        "cmd": [PYTHON, str(ROOT_DIR / "src" / "benchmark.py"), "--dataset", "real"],
    },
]


def run_command(step_name: str, cmd: list[str]) -> bool:
    print(f"\n{'='*70}")
    print(f"RUNNING: {step_name}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    try:
        # Inherits stdout/stderr to stream live progress bars and outputs
        res = subprocess.run(cmd, cwd=ROOT_DIR, check=True)
        elapsed = time.perf_counter() - t0
        print(f"✔ COMPLETED in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - t0
        print(f"\n✖ FAILED after {elapsed:.2f}s with exit code {e.returncode}", file=sys.stderr)
        return False
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(130)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Vector Database Pipeline Runner")
    parser.add_argument(
        "--launch-ui",
        action="store_true",
        help="Automatically launch the Streamlit visualizer after all pipeline steps pass",
    )
    args = parser.parse_args()

    total_start = time.perf_counter()

    for idx, step in enumerate(PIPELINE_STEPS, 1):
        success = run_command(step["name"], step["cmd"])
        if not success:
            print(f"\nPipeline halted at step {idx}/{len(PIPELINE_STEPS)}: {step['name']}")
            sys.exit(1)

    total_time = time.perf_counter() - total_start
    print(f"\n{'#'*70}")
    print(f"ALL PIPELINE STAGES PASSED SUCCESSFULLY in {total_time:.2f}s")
    print(f"{'#'*70}")
    print("Generated Artifacts in data/:")
    for f in (ROOT_DIR / "data").glob("*"):
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    if args.launch_ui:
        print("\nLaunching Streamlit Dashboard...")
        subprocess.run(["streamlit", "run", str(ROOT_DIR / "app.py")], cwd=ROOT_DIR)
    else:
        print("\nTo launch the visualizer now, run:")
        print("  uv run streamlit run app.py")


if __name__ == "__main__":
    main()