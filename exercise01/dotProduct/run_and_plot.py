"""
Minimal timing script for exercise 1.4 (dot product).

Usage (from exercise01/dotProduct):

  python run_and_plot.py

Assumes you already built the executable with:

  mkdir -p build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make

The executable is named dotProduct (from CMakeLists.txt).
"""

import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_DIR = SCRIPT_DIR / "build"
EXE = BUILD_DIR / "dotProduct"

# Vector sizes from the exercise sheet
DOT_DIMS = [10_000, 1_000_000, 10_000_000, 100_000_000]

# Number of times we run each configuration to average out noise
NUM_TIMING_RUNS = 3


def time_dot_product(dim: int, gpu_flag: int) -> float:
    """Return average time in seconds for testDotProduct <dim> <gpu_flag>."""
    times = []
    for _ in range(NUM_TIMING_RUNS):
        t0 = time.perf_counter()
        subprocess.run(
            [str(EXE), str(dim), str(gpu_flag)],
            cwd=BUILD_DIR,
            check=True,
        )
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def main() -> None:
    if not EXE.exists():
        raise SystemExit(f"Executable not found at {EXE}. Run: cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make")

    cpu_times = []
    gpu_times = []

    print("Timing dot product (exercise 1.4)...\n")
    for dim in DOT_DIMS:
        t_cpu = time_dot_product(dim, 0)
        t_gpu = time_dot_product(dim, 1)
        cpu_times.append(t_cpu)
        gpu_times.append(t_gpu)
        print(f"dim={dim:>11}: CPU {t_cpu:.4f} s   GPU {t_gpu:.4f} s")

    # Save CSV next to this script
    csv_path = SCRIPT_DIR / "dot_product_timings.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("exercise,config,dim,time_seconds\n")
        for dim, t_cpu, t_gpu in zip(DOT_DIMS, cpu_times, gpu_times):
            f.write(f"dotProduct,cpu,{dim},{t_cpu:.6f}\n")
            f.write(f"dotProduct,gpu,{dim},{t_gpu:.6f}\n")
    print(f"\nCSV saved to {csv_path}")

    # Plot CPU vs GPU timing (log–log)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(DOT_DIMS, cpu_times, "o-", label="CPU")
    ax.plot(DOT_DIMS, gpu_times, "s-", label="GPU")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Vector size")
    ax.set_ylabel("Time (s)")
    ax.set_title("Dot product (1.4, 100 iterations per run)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = SCRIPT_DIR / "dot_product_timings.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {png_path}")


if __name__ == "__main__":
    main()
