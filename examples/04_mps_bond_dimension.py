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
Example 4 — MPS GPU Mode with Bond Dimension Sweep
====================================================

Uses Matrix Product State (MPS) simulation on the GPU to study H₂O with a
larger active space, sweeping over bond dimensions to show the
accuracy-performance trade-off.

What is MPS simulation?
-----------------------
Standard statevector simulation stores all 2^N amplitudes — this limits us to
~25-30 qubits on a single GPU. MPS (Matrix Product State) is a *compressed*
representation that factorises the state into a chain of tensors connected by
"bonds". The bond dimension χ controls how much entanglement the MPS can
represent:

    χ = 1    → only product states (no entanglement)
    χ = 16   → limited entanglement
    χ = 256  → high-fidelity for most chemistry circuits
    χ → ∞    → exact statevector

For chemistry circuits (VQE ansatze with local entangling gates), MPS is
remarkably efficient — moderate bond dimensions often give near-exact results.
This lets Maestro simulate 50-100+ qubits on a single GPU.

Why this is unique to Maestro
-----------------------------
Qiskit does not offer GPU-accelerated MPS simulation. Maestro's MPS backend
runs entirely on the GPU via cuQuantum/cuTensorNet, giving you both large
qubit counts and GPU speed.

What this example shows
-----------------------
- H₂O molecule with a (4e, 4o) active space → 8 qubits
- VQE with MPS at χ = 16, 64, 256
- How accuracy improves with bond dimension
- How wall-clock time scales with χ

Usage
-----
    python 04_mps_bond_dimension.py
    python 04_mps_bond_dimension.py --gpu
"""

import argparse
import time

from pyscf import gto, scf, mcscf
from qoro_maestro_pyscf import MaestroSolver


def main():
    parser = argparse.ArgumentParser(description="MPS bond dimension sweep")
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  MPS GPU MODE — BOND DIMENSION SWEEP")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    # --- Molecule: H₂O ---
    mol = gto.M(
        atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
        basis="sto-3g",
        verbose=0,
    )
    hf_obj = scf.RHF(mol).run()

    norb = 4
    nelec = 4
    n_qubits = 2 * norb

    print(f"\n  Molecule        : H₂O (STO-3G)")
    print(f"  Active space    : ({nelec}e, {norb}o) → {n_qubits} qubits")
    print(f"  HF energy       : {hf_obj.e_tot:+.10f} Ha")

    # --- FCI baseline ---
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]
    print(f"  FCI energy      : {fci_e:+.10f} Ha")

    # --- MPS sweep ---
    print(f"\n  {'χ':>6s}  {'Energy':>14s}  {'Error (mHa)':>12s}  {'Time (s)':>10s}")
    print("  " + "-" * 48)

    for chi in [16, 64, 256]:
        cas_mps = mcscf.CASCI(hf_obj, norb, nelec)
        cas_mps.fcisolver = MaestroSolver(
            ansatz="hardware_efficient",
            ansatz_layers=2,
            backend=backend,
            simulation="mps",
            mps_bond_dim=chi,
            maxiter=100,
            verbose=False,
        )

        t0 = time.perf_counter()
        mps_e = cas_mps.kernel()[0]
        elapsed = time.perf_counter() - t0

        error = abs(mps_e - fci_e) * 1000
        print(f"  {chi:6d}  {mps_e:+14.8f}  {error:12.2f}  {elapsed:10.2f}")

    print()
    print("  Takeaway: Higher bond dimension → more accurate, but slower.")
    print("  For chemistry VQE circuits, χ=64 is often sufficient.")
    print("  Maestro's GPU MPS lets you push to 50-100+ qubits.")


if __name__ == "__main__":
    main()
