"""Reproduce the archived count experiments and run the local profiling tools."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

from .pilot import run_grid
from .robustness import reproduce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reproduction"))
    parser.add_argument("--robustness-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1, user_api="blas"):
        for seed in (20261113, 20261114):
            rows = list(run_grid(seed, trials=300))
            with (args.out / f"pilot-{seed}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    reproduce(args.out / "robustness")
    if not args.robustness_only:
        subprocess.run(
            [sys.executable, "-m", "music_benchmark.profile", "--reference-frames", "24",
             "--output-dir", str(args.out / "profile")],
            check=True,
        )
        from .profile_plot import plot
        plot(args.out / "profile")
        subprocess.run(
            [sys.executable, "-m", "music_benchmark.hardware", "--output-dir", str(args.out / "hardware")],
            check=True,
        )


if __name__ == "__main__":
    main()
