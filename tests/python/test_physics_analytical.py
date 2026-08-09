#!/usr/bin/env python3
"""
Analytical Physics Verification & C ABI Cross-Validation Suite for Shroud.
Validates closed-form physical equations against compiled Rust/C ABI kernels.

Copyright (c) 2026 Aneesh Krishna Parthasarathy
Licensed under the ISC License.
"""

import ctypes
import math
import os
import platform
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SPEED_OF_LIGHT = 299792458.0
PLANCK_H = 6.62607015e-34
BOLTZMANN_K = 1.380649e-23


def load_shroud_cdll():
    ext = "dylib" if platform.system() == "Darwin" else "so"
    for profile in ("debug", "release"):
        candidate = ROOT_DIR / "target" / profile / f"libshroud_c.{ext}"
        if candidate.exists():
            lib = ctypes.CDLL(str(candidate))
            lib.shroud_version_major.restype = ctypes.c_int
            lib.shroud_eval_henyey_greenstein.argtypes = [ctypes.c_float, ctypes.c_float]
            lib.shroud_eval_henyey_greenstein.restype = ctypes.c_float
            lib.shroud_eval_rayleigh_phase.argtypes = [ctypes.c_float]
            lib.shroud_eval_rayleigh_phase.restype = ctypes.c_float
            lib.shroud_eval_lwir_radiance.argtypes = [ctypes.c_float]
            lib.shroud_eval_lwir_radiance.restype = ctypes.c_float
            lib.shroud_calculate_lidar_photons.argtypes = [
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
            ]
            lib.shroud_calculate_lidar_photons.restype = ctypes.c_float
            lib.shroud_morton_3d.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
            lib.shroud_morton_3d.restype = ctypes.c_uint32
            return lib
    return None


class TestShroudAnalyticalPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_shroud_cdll()

    def test_henyey_greenstein_normalization_and_c_abi(self):
        for g in (-0.4, 0.0, 0.65, 0.85):
            steps = 2000
            d_mu = 2.0 / steps
            integral = 0.0
            for i in range(steps):
                mu = -1.0 + (i + 0.5) * d_mu
                denom = 1.0 + g * g - 2.0 * g * mu
                p = (1.0 - g * g) / (4.0 * math.pi * (denom ** 1.5))
                integral += 2.0 * math.pi * p * d_mu
                if self.lib is not None and i % 400 == 0:
                    c_val = self.lib.shroud_eval_henyey_greenstein(mu, g)
                    self.assertAlmostEqual(c_val, p, places=4)
            self.assertAlmostEqual(integral, 1.0, places=3)

    def test_rayleigh_phase_analytical_and_c_abi(self):
        for mu in (-1.0, -0.5, 0.0, 0.5, 1.0):
            expected = (3.0 / (16.0 * math.pi)) * (1.0 + mu * mu)
            if self.lib is not None:
                c_val = self.lib.shroud_eval_rayleigh_phase(mu)
                self.assertAlmostEqual(c_val, expected, places=5)

    def test_lwir_planck_radiance_quadrature(self):
        t_k = 300.0
        steps = 512
        lam_a, lam_b = 8.0e-6, 14.0e-6
        dl = (lam_b - lam_a) / steps
        total = 0.0
        for i in range(steps):
            lam = lam_a + (i + 0.5) * dl
            expo = (PLANCK_H * SPEED_OF_LIGHT) / (lam * BOLTZMANN_K * t_k)
            rad = (2.0 * PLANCK_H * SPEED_OF_LIGHT * SPEED_OF_LIGHT) / (
                (math.exp(expo) - 1.0) * (lam ** 5)
            )
            total += rad * dl
        self.assertGreater(total, 50.0)
        self.assertLess(total, 60.0)
        if self.lib is not None:
            c_lwir = self.lib.shroud_eval_lwir_radiance(t_k)
            self.assertAlmostEqual(c_lwir, total, delta=0.1)

    def test_fmcw_1550nm_high_doppler_inversion(self):
        f_center = 193.4e12
        bw = 600.0e6
        t_c = 25.0e-6
        lam = SPEED_OF_LIGHT / f_center

        for r_true, v_true in ((25.0, 12.5), (120.0, -28.0), (210.0, 0.0)):
            f_range = (2.0 * bw * r_true) / (SPEED_OF_LIGHT * t_c)
            f_doppler = (2.0 * v_true) / lam
            f_up = f_range - f_doppler
            f_down = f_range + f_doppler
            r_est = (SPEED_OF_LIGHT * t_c * 0.5 * (f_down + f_up)) / (2.0 * bw)
            v_est = (lam * 0.5 * (f_down - f_up)) / 2.0
            self.assertAlmostEqual(r_est, r_true, places=4)
            self.assertAlmostEqual(v_est, v_true, places=4)

    def test_morton_3d_bit_interleaving(self):
        if self.lib is not None:
            m_zero = self.lib.shroud_morton_3d(0.0, 0.0, 0.0)
            m_mid = self.lib.shroud_morton_3d(0.5, 0.5, 0.5)
            m_one = self.lib.shroud_morton_3d(1.0, 1.0, 1.0)
            self.assertEqual(m_zero, 0)
            self.assertLess(m_zero, m_mid)
            self.assertLess(m_mid, m_one)
            self.assertEqual(m_one, 0x3FFFFFFF)


if __name__ == "__main__":
    unittest.main(verbosity=2)
