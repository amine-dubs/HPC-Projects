"""Pre-warm and JIT-compile Numba kernels to avoid compilation overhead during benchmarking.

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from fractal_core import render_block


def warmup():
    print("Warming up serial kernel...")
    _ = render_block(0, 10, 0, 10, -2.0, 1.0, 0.01, 0.01, 50)
    print("Warm-up complete.")


if __name__ == "__main__":
    warmup()

