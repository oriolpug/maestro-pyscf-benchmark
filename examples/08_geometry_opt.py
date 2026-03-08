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
# along with qoro-maestro-pyscf. If not, see <https://www.gnu.org/licenses/>.

"""
Example 9 — Geometry Optimisation
===================================

Optimises the bond length of H₂ by minimising the CASCI energy as a
function of nuclear coordinates.

Why geometry optimisation?
--------------------------
A single-point energy calculation tells you the energy at ONE geometry.
But chemists need equilibrium structures, transition states, and reaction
paths. Geometry optimisation finds the nuclear coordinates that minimise
the total energy — this gives the equilibrium bond length.

PySCF's `geomopt` module handles the nuclear gradient loop (using finite
differences or analytical gradients). At each nuclear geometry, it calls
CASCI with our MaestroSolver to get the electronic energy. The result is
the GPU-accelerated equilibrium geometry.

What this example shows
-----------------------
- Starting from a stretched H₂ (d=1.2 Å), find the equilibrium geometry
- PySCF drives the nuclear optimisation; Maestro solves the electronic part
- Compare the optimised bond length with the known value (~0.74 Å)

Usage
-----
    python 09_geometry_opt.py
    python 09_geometry_opt.py --gpu
"""

import argparse
import time

import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.geomopt.geometric_solver import optimize as geo_optimize

from qoro_maestro_pyscf import MaestroSolver


def make_casci(mol, backend):
    """Build a CASCI object with MaestroSolver for the given molecule."""
    hf_obj = scf.RHF(mol).run()
    cas = mcscf.CASCI(hf_obj, 2, 2)
    cas.fcisolver = MaestroSolver(
        ansatz="hardware_efficient",
        ansatz_layers=2,
        backend=backend,
        maxiter=100,
        verbose=False,
    )
    return cas


def main():
    parser = argparse.ArgumentParser(description="Geometry optimisation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  GEOMETRY OPTIMISATION — H₂")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    # Start from a stretched geometry
    initial_d = 1.2  # Å (equilibrium is ~0.74 Å)
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {initial_d}",
        basis="sto-3g",
        verbose=0,
        unit="angstrom",
    )

    print(f"\n  Initial bond length : {initial_d:.3f} Å")
    print(f"  Expected equilibrium: ~0.74 Å")

    # Build CASCI with MaestroSolver
    cas = make_casci(mol, backend)

    # Run geometry optimisation
    print(f"\n  Optimising nuclear coordinates...")
    t0 = time.perf_counter()

    try:
        mol_eq = geo_optimize(cas, verbose=0)
        elapsed = time.perf_counter() - t0

        # Extract optimised bond length
        coords = mol_eq.atom_coords(unit="angstrom")
        opt_d = np.linalg.norm(coords[1] - coords[0])

        print(f"  Optimisation complete in {elapsed:.1f}s")
        print(f"\n  {'─' * 50}")
        print(f"  Initial bond length  : {initial_d:.4f} Å")
        print(f"  Optimised bond length: {opt_d:.4f} Å")
        print(f"  Literature value     : 0.7414 Å")
        print(f"  Error                : {abs(opt_d - 0.7414):.4f} Å")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\n  Note: Geometry optimisation requires the 'geometric'")
        print(f"  package. Install with: pip install geometric")
        print(f"\n  Falling back to manual PES scan...")

        # Manual scan as fallback
        bond_lengths = np.arange(0.5, 1.5, 0.05)
        energies = []

        for d in bond_lengths:
            mol_d = gto.M(
                atom=f"H 0 0 0; H 0 0 {d}",
                basis="sto-3g",
                verbose=0,
                unit="angstrom",
            )
            cas_d = make_casci(mol_d, backend)
            e = cas_d.kernel()[0]
            energies.append(e)

        # Find minimum
        min_idx = np.argmin(energies)
        opt_d = bond_lengths[min_idx]

        print(f"\n  {'─' * 50}")
        print(f"  Minimum energy at   : {opt_d:.3f} Å")
        print(f"  Energy              : {energies[min_idx]:+.8f} Ha")
        print(f"  Literature value    : 0.741 Å")


if __name__ == "__main__":
    main()
