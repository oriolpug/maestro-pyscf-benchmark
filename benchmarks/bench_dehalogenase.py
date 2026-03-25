#!/usr/bin/env python3
# Copyright 2026 Qoro Quantum Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Dehalogenase SN2 Reaction Benchmark
=====================================

Benchmarks Maestro VQE on the dehalogenase enzyme reaction from the Qrunch
tutorial (qrunch_tutorials/dehalogenase-tutorial).

The Qrunch tutorial uses projective embedding (DFT environment + MP2 embedded
region via qrunch).  Here we use the bare 5-atom embedded subsystem directly
in PySCF — same active space, no environmental embedding — which lets us run
the benchmark without the qrunch package and use MaestroSolver as the solver.

System
------
  Embedded region : 5 atoms — C, O, O, Cl, C (reaction-center atoms, 0-based
                    indices [5, 6, 7, 18, 19] in the 27-atom small model)
  Basis           : STO-3G
  Charge          : -1  (same as tutorial)
  Active space    : CAS(10e, 10o) = 20 qubits  →  MPS simulation
  Reaction frames : 11 frames along the SN2 path (Frame 0 → Frame 10)

  The 5-atom fragment has 46 electrons (45 nuclear + 1 extra for charge -1),
  29 STO-3G basis functions, so 6 virtual MOs.  CASCI(10,10) spans the last 5
  occupied and first 5 virtual MOs.  Classical FCI in this space has dimension
  C(10,5)² ≈ 63 000 — trivial for PySCF and a clean reference.

Comparison
----------
  Classical CASCI/FCI  : PySCF default FCI solver (exact in the active space)
  Maestro UpCCD        : MPS simulation, χ configurable (default 64)

Runtime notes
-------------
  Each frame: one MPS VQE (20 qubits, χ=64, 50 iters) ≈ 5–15 min on CPU.
    3 frames  (default: 0, 5, 10)  :  ~15–45 min
    11 frames (--frames all)       :  ~1–3 hours
  Use --chi 32 for ~4× speed-up at some accuracy cost.
  Use --gpu if a GPU is available.

Usage
-----
    poetry run python benchmarks/bench_dehalogenase.py              # frames 0,5,10
    poetry run python benchmarks/bench_dehalogenase.py --frames all # all 11 frames
    poetry run python benchmarks/bench_dehalogenase.py --chi 32     # faster
    poetry run python benchmarks/bench_dehalogenase.py --gpu
"""

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from pyscf import gto, mcscf, scf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qoro_maestro_pyscf import MaestroSolver

CACHE_DIR = Path(__file__).parent / "cache"

# Path to the Qrunch tutorial geometry
DEHALOGENASE_XYZ = (
    ROOT / "qrunch_tutorials/dehalogenase-tutorial/data/dehalogenase_reaction_small.xyz"
)
# 0-based atom indices of the embedded reaction-center atoms: C, O, O, Cl, C
EMBEDDED_ATOM_INDICES = [5, 6, 7, 18, 19]

# Active space matching the Qrunch tutorial
NORB  = 10   # 10 spatial orbitals → 20 qubits
NELEC = (5, 5)  # 10 electrons, restricted (5α + 5β)

# Statevector limit from run_benchmarks.py; 20q > 14q so MPS is used
SV_QUBIT_LIMIT = 14


# ── JSON encoder ───────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ── XYZ parsing ───────────────────────────────────────────────────────────────

def _parse_xyz_frames(
    path: Path,
    atom_indices: list[int] | None = None,
) -> list[list[tuple[str, float, float, float]]]:
    """Parse a multi-frame XYZ file.

    Returns a list of frames; each frame is a list of (element, x, y, z).
    If atom_indices is given, only those atoms (0-based) are kept per frame.
    """
    frames = []
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue
        i += 2  # skip n_atoms line and comment/frame-label line
        atoms: list[tuple[str, float, float, float]] = []
        for _ in range(n_atoms):
            parts = lines[i].split()
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
            i += 1
        if atom_indices is not None:
            atoms = [atoms[k] for k in atom_indices]
        frames.append(atoms)
    return frames


def _mol_from_atoms(
    atoms: list[tuple[str, float, float, float]],
    charge: int = 0,
    spin: int = 0,
    basis: str = "sto-3g",
) -> gto.Mole:
    """Build a PySCF Mole from a list of (element, x, y, z) tuples (Angstrom)."""
    atom_str = "; ".join(f"{e} {x:.8f} {y:.8f} {z:.8f}" for e, x, y, z in atoms)
    return gto.M(atom=atom_str, basis=basis, charge=charge, spin=spin,
                 verbose=0, unit="Angstrom")


# ── Low-level runners ─────────────────────────────────────────────────────────

def _run_hf(mol: gto.Mole) -> tuple[scf.hf.SCF, float]:
    """Returns (hf, wall_time_s)."""
    t0 = time.perf_counter()
    hf = scf.RHF(mol)
    hf.verbose = 0
    hf.run()
    return hf, time.perf_counter() - t0


def _run_casci_fci(hf: scf.hf.SCF, norb: int, nelec: tuple[int, int],
                   timeout_s: int = 300) -> tuple[float | None, float | None]:
    """CASCI with PySCF FCI (exact in active space). Returns (energy, wall_time)."""
    import signal

    def _handler(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        t0 = time.perf_counter()
        cas = mcscf.CASCI(hf, norb, nelec)
        cas.verbose = 0
        energy = cas.kernel()[0]
        return energy, time.perf_counter() - t0
    except TimeoutError:
        return None, None
    finally:
        signal.alarm(0)


def _run_maestro(hf: scf.hf.SCF, norb: int, nelec: tuple[int, int],
                 ansatz: str, backend: str, mps_bond_dim: int = 64,
                 **kwargs) -> dict:
    """Run Maestro VQE.  20q > SV_QUBIT_LIMIT → always MPS."""
    n_qubits = 2 * norb
    simulation = "statevector" if n_qubits <= SV_QUBIT_LIMIT else "mps"
    cas = mcscf.CASCI(hf, norb, nelec)
    cas.verbose = 0
    kw = dict(ansatz=ansatz, backend=backend, simulation=simulation,
              verbose=False, **kwargs)
    if simulation == "mps":
        kw["mps_bond_dim"] = mps_bond_dim
    cas.fcisolver = MaestroSolver(**kw)
    try:
        t0 = time.perf_counter()
        energy = cas.kernel()[0]
        elapsed = time.perf_counter() - t0
        return {
            "status": "ok",
            "energy": energy,
            "time": elapsed,
            "simulation": simulation,
            "mps_bond_dim": mps_bond_dim if simulation == "mps" else None,
            "converged": cas.fcisolver.converged,
            "iters": len(cas.fcisolver.energy_history),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rel_energy_ha(e, e_ref):
    """Energy relative to e_ref in Ha (reaction profile)."""
    if e is None or e_ref is None:
        return None
    return e - e_ref


# ── Main benchmark ────────────────────────────────────────────────────────────

DEFAULT_FRAMES = [0, 5, 10]  # reactant, ~TS, product


def bench_dehalogenase(
    gpu: bool = False,
    frame_indices: list[int] | None = None,
    norb: int = NORB,
    nelec: tuple[int, int] = NELEC,
    mps_bond_dim: int = 64,
    ansatz: str = "upccd",
    maxiter: int = 50,
) -> dict:
    """Dehalogenase SN2 reaction — Maestro UpCCD MPS vs. classical CASCI/FCI.

    Runs across the reaction coordinate frames, recording HF, FCI, and Maestro
    energies.  Relative energies (meV, frame 0 = 0) give the activation barrier.
    """
    if not DEHALOGENASE_XYZ.exists():
        raise FileNotFoundError(
            f"Geometry file not found: {DEHALOGENASE_XYZ}\n"
            "Make sure the qrunch_tutorials submodule is initialised."
        )

    all_frames = _parse_xyz_frames(DEHALOGENASE_XYZ, EMBEDDED_ATOM_INDICES)
    if frame_indices is None:
        frame_indices = DEFAULT_FRAMES

    selected = [(i, all_frames[i]) for i in frame_indices if i < len(all_frames)]
    n_qubits = 2 * norb

    print(f"\nDehalogenase SN2  CAS({sum(nelec)}e,{norb}o) = {n_qubits}q  "
          f"MPS χ={mps_bond_dim}  ansatz={ansatz}")
    print(f"  Embedded subsystem: 5 atoms (C,O,O,Cl,C)  STO-3G  charge=-1")
    print(f"  Frames: {frame_indices}")
    print(f"  err  = E_Maestro - E_FCI  (VQE variational error; FCI is exact in active space)")
    print(f"  ΔE   = E(frame) - E(frame 0)  (reaction energy profile, barrier height)")
    print(f"  {'Frame':>5}  {'HF (Ha)':>13} {'t':>5}  "
          f"{'FCI (Ha)':>13} {'t':>5}  "
          f"{'Maestro (Ha)':>13} {'t':>7}  "
          f"{'err (Ha)':>13}  "
          f"{'ΔE_FCI (Ha)':>12}  {'ΔE_M (Ha)':>12}")

    records = []
    e_fci_frame0 = None
    e_m_frame0   = None

    for frame_idx, atoms in selected:
        mol = _mol_from_atoms(atoms, charge=-1, spin=0, basis="sto-3g")
        hf, t_hf = _run_hf(mol)

        print(f"  {frame_idx:5d}  running FCI...", end="", flush=True)
        e_fci, t_fci = _run_casci_fci(hf, norb, nelec)

        print(f"  running Maestro...", end="", flush=True)
        m = _run_maestro(hf, norb, nelec, ansatz,
                         "gpu" if gpu else "cpu",
                         mps_bond_dim=mps_bond_dim, maxiter=maxiter)
        e_m   = m.get("energy")
        t_m   = m.get("time")

        # Store frame-0 reference energies for relative profile
        if e_fci_frame0 is None and e_fci is not None:
            e_fci_frame0 = e_fci
        if e_m_frame0 is None and e_m is not None:
            e_m_frame0 = e_m

        def _s(e): return f"{e:+13.6f}" if e is not None else "         N/A "
        def _t(t): return f"{t:5.1f}s" if t is not None else "   N/A"
        def _sd(e, ref):
            v = _rel_energy_ha(e, ref)
            return f"{v:+12.6f}" if v is not None else "         N/A "

        err_ha = (e_m - e_fci) if (e_m is not None and e_fci is not None) else None

        print(
            f"\r  {frame_idx:5d}  {_s(hf.e_tot)} {_t(t_hf)}  "
            f"{_s(e_fci)} {_t(t_fci)}  "
            f"{_s(e_m)} {_t(t_m):>7}  "
            f"{_s(err_ha)}  "
            f"{_sd(e_fci, e_fci_frame0):>12}  "
            f"{_sd(e_m, e_m_frame0):>12}"
        )

        records.append({
            "frame": frame_idx,
            "e_hf": hf.e_tot, "t_hf": t_hf,
            "e_fci": e_fci,   "t_fci": t_fci,
            "maestro": m,
            "err_maestro_ha": err_ha,
            "delta_e_fci_ha": _rel_energy_ha(e_fci, e_fci_frame0),
            "delta_e_maestro_ha": _rel_energy_ha(e_m, e_m_frame0),
        })

    return {
        "name": "dehalogenase",
        "molecule": "dehalogenase_embedded_5atom",
        "basis": "sto-3g",
        "charge": -1,
        "norb": norb,
        "nelec": list(nelec),
        "n_qubits": n_qubits,
        "ansatz": ansatz,
        "mps_bond_dim": mps_bond_dim,
        "frame_indices": frame_indices,
        "records": records,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dehalogenase SN2 reaction benchmark — Maestro vs. CASCI/FCI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  poetry run python benchmarks/bench_dehalogenase.py                 # frames 0,5,10 (default)
  poetry run python benchmarks/bench_dehalogenase.py --frames all    # all 11 frames (~1-3h)
  poetry run python benchmarks/bench_dehalogenase.py --frames 0,5,10 # explicit 3 frames
  poetry run python benchmarks/bench_dehalogenase.py --chi 32        # ~4x faster, less accurate
  poetry run python benchmarks/bench_dehalogenase.py --gpu           # GPU backend
        """,
    )
    parser.add_argument("--gpu",     action="store_true", help="Maestro GPU backend")
    parser.add_argument("--chi",     type=int, default=64,
                        help="MPS bond dimension χ (default: 64)")
    parser.add_argument("--frames",  type=str, default=None,
                        help="Comma-separated frame indices or 'all'; "
                             "e.g. 0,5,10  (default: 0,5,10)")
    parser.add_argument("--ansatz",  type=str, default="upccd",
                        choices=["upccd", "hardware_efficient"],
                        help="VQE ansatz (default: upccd)")
    parser.add_argument("--maxiter", type=int, default=50,
                        help="Max VQE iterations per frame (default: 50)")
    parser.add_argument("--output",  type=str, default=None,
                        help="Override output JSON path")
    args = parser.parse_args()

    if args.frames is None:
        frame_indices = None  # uses DEFAULT_FRAMES inside bench_dehalogenase
    elif args.frames.strip().lower() == "all":
        frame_indices = list(range(11))
    else:
        frame_indices = [int(x) for x in args.frames.split(",")]

    print("=" * 72)
    print("  DEHALOGENASE SN2 BENCHMARK  —  Maestro vs. CASCI/FCI")
    print(f"  GPU     : {'enabled' if args.gpu else 'disabled'}")
    print(f"  χ       : {args.chi}")
    print(f"  Ansatz  : {args.ansatz}")
    print(f"  Frames  : {frame_indices or DEFAULT_FRAMES}")
    print(f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    t0 = time.perf_counter()
    result = bench_dehalogenase(
        gpu=args.gpu,
        frame_indices=frame_indices,
        mps_bond_dim=args.chi,
        ansatz=args.ansatz,
        maxiter=args.maxiter,
    )
    total_time = time.perf_counter() - t0

    output = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "gpu": args.gpu,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "total_time_s": round(total_time, 2),
        },
        "benchmark": result,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (
        CACHE_DIR / f"dehalogenase_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)

    print(f"\n{'=' * 72}")
    print(f"  Done in {total_time:.1f}s")
    print(f"  Results : {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
