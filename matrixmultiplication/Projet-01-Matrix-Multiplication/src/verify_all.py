"""
Comprehensive Verification and Test Suite
High-Performance Computing (HPC) Project: Matrix Multiplication

This script runs automated tests validating:
1. Matrix generation and seed determinism
2. Numerical correctness of Serial Baseline against NumPy BLAS
3. Numerical correctness of Multi-Threaded Naive kernel against NumPy BLAS
4. Numerical correctness of Multi-Threaded Cache-Blocked kernel against NumPy BLAS
5. Correctness of GFLOP/s and memory calculation routines
6. Dynamic thread count adjustment under Numba
"""

import os
import sys
import unittest
import numpy as np
import numba

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import generate_matrices, compute_reference, verify_result, calculate_gflops, get_matrix_memory_mb
from src.baseline_serial import matmul_serial_ikj, matmul_serial_ijk, warmup_serial
from src.version1_threads import matmul_threads_naive, matmul_threads_blocked, warmup_threads


class TestMatrixMultiplication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n[SETUP] Compiling Numba JIT functions for test suite...")
        warmup_serial(n_warmup=32)
        warmup_threads(n_warmup=32)

    def test_01_matrix_generation(self):
        """Test matrix generation determinism and data types."""
        A1, B1 = generate_matrices(128, seed=42)
        A2, B2 = generate_matrices(128, seed=42)
        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(B1, B2)
        self.assertEqual(A1.dtype, np.float64)
        self.assertEqual(A1.shape, (128, 128))

    def test_02_serial_ikj_correctness(self):
        """Test numerical correctness of Serial (i, k, j) kernel."""
        n = 128
        A, B = generate_matrices(n, seed=10)
        C_ref = compute_reference(A, B)
        C_test = np.zeros((n, n), dtype=np.float64)
        matmul_serial_ikj(A, B, C_test)
        is_ok, max_abs, max_rel = verify_result(C_test, C_ref)
        self.assertTrue(is_ok, f"Serial IKJ failed with max_abs={max_abs}")
        self.assertLess(max_abs, 1e-8)

    def test_03_serial_ijk_correctness(self):
        """Test numerical correctness of Naive Serial (i, j, k) kernel."""
        n = 128
        A, B = generate_matrices(n, seed=11)
        C_ref = compute_reference(A, B)
        C_test = np.zeros((n, n), dtype=np.float64)
        matmul_serial_ijk(A, B, C_test)
        is_ok, max_abs, max_rel = verify_result(C_test, C_ref)
        self.assertTrue(is_ok, f"Serial IJK failed with max_abs={max_abs}")
        self.assertLess(max_abs, 1e-8)

    def test_04_threads_naive_correctness(self):
        """Test numerical correctness of Multi-Threaded Naive kernel."""
        n = 256
        A, B = generate_matrices(n, seed=12)
        C_ref = compute_reference(A, B)
        for p in [1, 2, 4]:
            if p > os.cpu_count():
                continue
            numba.set_num_threads(p)
            C_test = np.zeros((n, n), dtype=np.float64)
            matmul_threads_naive(A, B, C_test)
            is_ok, max_abs, max_rel = verify_result(C_test, C_ref)
            self.assertTrue(is_ok, f"Threaded naive (p={p}) failed with max_abs={max_abs}")
            self.assertLess(max_abs, 1e-8)

    def test_05_threads_blocked_correctness(self):
        """Test numerical correctness of Cache-Blocked kernel."""
        n = 256
        A, B = generate_matrices(n, seed=13)
        C_ref = compute_reference(A, B)
        for bsz in [16, 32, 64]:
            numba.set_num_threads(2)
            C_test = np.zeros((n, n), dtype=np.float64)
            matmul_threads_blocked(A, B, C_test, block_size=bsz)
            is_ok, max_abs, max_rel = verify_result(C_test, C_ref)
            self.assertTrue(is_ok, f"Threaded blocked (bsz={bsz}) failed with max_abs={max_abs}")
            self.assertLess(max_abs, 1e-8)

    def test_06_metrics_calculation(self):
        """Test FLOP/s and Memory helper formulas."""
        # For n=1000, 2*n^3 = 2e9 FLOPs. In 1.0 second -> 2.0 GFLOP/s
        gflops = calculate_gflops(1000, 1.0)
        self.assertAlmostEqual(gflops, 2.0, places=4)

        # Memory for 1024x1024 float64: 1024*1024*8 = 8MB
        mem = get_matrix_memory_mb(1024, dtype=np.float64)
        self.assertAlmostEqual(mem, 8.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
