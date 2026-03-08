#!/usr/bin/env python3
# This file is part of qoro-maestro-pyscf.
#
# Copyright (C) 2026 Qoro Quantum Ltd.
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
Example 3 — GPU vs CPU Benchmark
=================================

Runs the same VQE calculation on both GPU and CPU backends and compares
wall-clock performance.

Why GPU matters for VQE
-----------------------
A single VQE iteration requires evaluating ⟨ψ(θ)|H|ψ(θ)⟩, which involves:

  1. Preparing |ψ(θ)⟩ — applying the parameterised circuit
  2. Computing expectation values for each Pauli term in the Hamiltonian

For a molecule with N qubits, the statevector has 2^N amplitudes, and each
gate is a 2^N × 2^N matrix multiply (or sparse equivalent). On a GPU, these
are massively parallelised — thousands of amplitudes updated simultaneously.

A typical VQE runs 100–500 iterations, each with dozens of Pauli evaluations.
The GPU speedup compounds across all of these.

What Maestro offers
-------------------
- **Same API** — switching from CPU to GPU is a single argument change
- **Automatic fallback** — if no GPU is available, falls back gracefully to CPU
- **No code changes** — your circuit, ansatz, and optimiser are identical

What this example shows
-----------------------
- Identical VQE on H₂ with both CPU and GPU backends
- Side-by-side timing comparison
- Both yield the same energy (within optimiser tolerance)

Usage
-----
    python 03_gpu_vs_cpu.py
"""

import time

from pyscf import gto, scf, mcscf
from qoro_maestro_pyscf import MaestroSolver


def main():
    print("=" * 72)
    print("  GPU vs CPU BENCHMARK")
    print("=" * 72)

    mol = gto.M(atom="H 0 0 0; H 0 0 0.7414", basis="sto-3g", verbose=0)
    hf_obj = scf.RHF(mol).run()

    # FCI baseline
    cas_fci = mcscf.CASCI(hf_obj, 2, 2)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]
    print(f"\n  FCI energy (exact): {fci_e:+.10f} Ha\n")

    results = {}

    for backend_name in ["cpu", "gpu"]:
        print(f"  Running on {backend_name.upper()}...")

        cas = mcscf.CASCI(hf_obj, 2, 2)
        cas.fcisolver = MaestroSolver(
            ansatz="hardware_efficient",
            ansatz_layers=2,
            backend=backend_name,
            maxiter=100,
            verbose=False,
        )

        t0 = time.perf_counter()
        e = cas.kernel()[0]
        elapsed = time.perf_counter() - t0

        error = abs(e - fci_e) * 1000
        iters = len(cas.fcisolver.energy_history)

        results[backend_name] = {"energy": e, "time": elapsed, "iters": iters}

        print(f"    Energy  : {e:+.10f} Ha  (err: {error:.2f} mHa)")
        print(f"    Time    : {elapsed:.3f} s")
        print(f"    Iters   : {iters}")
        print()

    # Comparison
    if "gpu" in results and "cpu" in results:
        speedup = results["cpu"]["time"] / max(results["gpu"]["time"], 1e-9)
        print(f"  {'─' * 50}")
        print(f"  GPU speedup   : {speedup:.1f}×")
        print(f"  Energy match  : {'✓ YES' if abs(results['gpu']['energy'] - results['cpu']['energy']) < 1e-4 else '✗ NO'}")
        print()
        print("  Note: Speedup increases with circuit size. For 4-qubit H₂,")
        print("  GPU overhead may dominate. Try larger molecules for dramatic gains.")


if __name__ == "__main__":
    main()
