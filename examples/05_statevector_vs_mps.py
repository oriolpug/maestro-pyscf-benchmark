#!/usr/bin/env python3
# This file is part of qoro-maestro-pyscf.
#
# Copyright (C) 2024 Qoro Quantum GmbH
#
# qoro-maestro-pyscf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# qoro-maestro-pyscf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with qoro-maestro-pyscf. If not, see <https://www.gnu.org/licenses/\>.

"""
Example 5 — Statevector vs MPS Accuracy Comparison
====================================================

Compares exact statevector simulation with approximate MPS simulation on the
same VQE problem, showing how MPS converges to the statevector result as
bond dimension increases.

When to use which?
------------------
+------------------+----------------------------+------------------------------+
| Mode             | Statevector                | MPS                          |
+------------------+----------------------------+------------------------------+
| Qubits           | ≤ 25-30                    | Up to 100+                   |
| Accuracy         | Exact                      | Controlled by χ              |
| Memory           | O(2^N)                     | O(N · χ²)                    |
| Best for         | Small molecules, debugging | Larger active spaces         |
| GPU acceleration | Yes                        | Yes (Maestro exclusive)      |
+------------------+----------------------------+------------------------------+

For small molecules (like H₂ with 4 qubits), statevector is fastest and exact.
For larger active spaces (8+ qubits), MPS gives you a knob to trade accuracy
for reachability.

What this example shows
-----------------------
- H₂ VQE with statevector, MPS χ=16, and MPS χ=64
- All three converge to the same energy for this small system
- Timing differences (statevector is faster for small systems)
- How to choose the right backend for your problem size

Usage
-----
    python 05_statevector_vs_mps.py
    python 05_statevector_vs_mps.py --gpu
"""

import argparse
import time

from pyscf import gto, scf, mcscf
from qoro_maestro_pyscf import MaestroSolver


def main():
    parser = argparse.ArgumentParser(description="Statevector vs MPS comparison")
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  STATEVECTOR vs MPS — ACCURACY COMPARISON")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    mol = gto.M(atom="H 0 0 0; H 0 0 0.7414", basis="sto-3g", verbose=0)
    hf_obj = scf.RHF(mol).run()

    # FCI baseline
    cas_fci = mcscf.CASCI(hf_obj, 2, 2)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]
    print(f"\n  FCI (exact)     : {fci_e:+.10f} Ha\n")

    # Configurations to compare
    configs = [
        ("Statevector", "statevector", None),
        ("MPS  χ=16", "mps", 16),
        ("MPS  χ=64", "mps", 64),
    ]

    print(f"  {'Mode':>16s}  {'Energy':>14s}  {'Error (mHa)':>12s}  {'Time (s)':>10s}")
    print("  " + "-" * 56)

    for label, sim_mode, chi in configs:
        solver_kwargs = dict(
            ansatz="hardware_efficient",
            ansatz_layers=2,
            backend=backend,
            simulation=sim_mode,
            maxiter=100,
            verbose=False,
        )
        if chi is not None:
            solver_kwargs["mps_bond_dim"] = chi

        cas = mcscf.CASCI(hf_obj, 2, 2)
        cas.fcisolver = MaestroSolver(**solver_kwargs)

        t0 = time.perf_counter()
        e = cas.kernel()[0]
        elapsed = time.perf_counter() - t0

        error = abs(e - fci_e) * 1000
        print(f"  {label:>16s}  {e:+14.8f}  {error:12.2f}  {elapsed:10.2f}")

    print()
    print("  For 4 qubits, all modes give the same result.")
    print("  MPS shows its value at 8+ qubits where statevector")
    print("  runs out of GPU memory.")


if __name__ == "__main__":
    main()
