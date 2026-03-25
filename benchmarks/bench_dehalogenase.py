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

Replicates the Kvantify Qrunch tutorial (qrunch_tutorials/dehalogenase-tutorial)
and adds FCI and Maestro for direct comparison.  All methods run on the same
problem (same geometry, same active space) and are reported side-by-side per
reaction frame:

  HF           — restricted Hartree-Fock (bare 5-atom subsystem, PySCF)
  FCI          — CASCI exact FCI in the active space (bare PySCF)
  Maestro      — UpCCD MPS VQE (bare PySCF + MaestroSolver)
  Qrunch CI    — FCI on the Qrunch-embedded Hamiltonian
                 (DFT env + MP2 orbitals + Pipek-Mezey + Manby projector)
  Qrunch VQE   — FAST-VQE (adaptive excitation-gate) on the embedded Hamiltonian

Note on comparability
---------------------
  FCI and Maestro use the bare 5-atom embedded subsystem (no environment).
  Qrunch CI and Qrunch VQE include the full 27-atom environment via projective
  embedding.  Absolute energies differ between the two approaches; however the
  reaction energy profiles ΔE (relative to frame 0) and the VQE errors
  (E_VQE − E_exact) are directly comparable across methods.

  Qrunch is run frame-by-frame via ground_state().projective_embedding() so
  that per-frame timings are reported.  Embedding results are cached per frame
  in benchmarks/cache/dehalogenase_qrunch/frame_N/ so the ~25-min setup is
  paid only on the first run.

  err_M   = E_Maestro  − E_FCI       (Maestro variational error)
  err_VQE = E_FAST-VQE − E_Qrunch_CI (Qrunch VQE variational error)
  ΔE      = E(frame)   − E(frame 0)  (reaction energy profile)

Qrunch columns show N/A when qrunch is not installed or the license is missing.

Runtime notes
-------------
  Maestro MPS (20q, χ=64, 50 iters) ≈ 5–15 min/frame CPU.
  Qrunch per frame: ~25 min first run (embedding setup, cached afterwards).
    3 frames  (default: 0, 5, 10) :  ~15–45 min  [Maestro] + ~75 min  [Qrunch first run]
    11 frames (--frames all)      :  ~1–3 hours  [Maestro] + ~4 hours [Qrunch first run]

Usage
-----
    poetry run python benchmarks/bench_dehalogenase.py
    poetry run python benchmarks/bench_dehalogenase.py --frames all
    poetry run python benchmarks/bench_dehalogenase.py --chi 32
    poetry run python benchmarks/bench_dehalogenase.py --no-qrunch
    poetry run python benchmarks/bench_dehalogenase.py --gpu
"""

import argparse
import json
import platform
import signal
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from pyscf import gto, mcscf, scf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qoro_maestro_pyscf import MaestroSolver

CACHE_DIR    = Path(__file__).parent / "cache"
QRUNCH_LICENSE = ROOT / "benchmarks/qrunch/license.txt"
QRUNCH_CACHE   = CACHE_DIR / "dehalogenase_qrunch"

DEHALOGENASE_XYZ = (
    ROOT / "benchmarks/geometries/dehalogenase_data/dehalogenase_reaction_small.xyz"
)
EMBEDDED_ATOM_INDICES = [5, 6, 7, 18, 19]  # C, O, O, Cl, C (0-based, 27-atom model)

NORB  = 10
NELEC = (5, 5)
SV_QUBIT_LIMIT = 14

DEFAULT_FRAMES = [0, 5, 10]


# ── JSON encoder ───────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ── XYZ I/O ───────────────────────────────────────────────────────────────────

def _parse_xyz_frames(
    path: Path,
    atom_indices: list[int] | None = None,
) -> list[list[tuple[str, float, float, float]]]:
    """Parse a multi-frame XYZ.  atom_indices selects a subset (0-based)."""
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
        i += 2  # skip count line + comment
        atoms: list[tuple[str, float, float, float]] = []
        for _ in range(n_atoms):
            parts = lines[i].split()
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
            i += 1
        if atom_indices is not None:
            atoms = [atoms[k] for k in atom_indices]
        frames.append(atoms)
    return frames


def _write_xyz(atoms: list[tuple[str, float, float, float]], path: Path,
               comment: str = "") -> None:
    """Write a single-frame XYZ file."""
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for e, x, y, z in atoms:
            f.write(f"{e}  {x:.8f}  {y:.8f}  {z:.8f}\n")


def _mol_from_atoms(atoms, charge=0, spin=0, basis="sto-3g"):
    atom_str = "; ".join(f"{e} {x:.8f} {y:.8f} {z:.8f}" for e, x, y, z in atoms)
    return gto.M(atom=atom_str, basis=basis, charge=charge, spin=spin,
                 verbose=0, unit="Angstrom")


# ── PySCF runners ─────────────────────────────────────────────────────────────

def _run_hf(mol) -> tuple:
    t0 = time.perf_counter()
    hf = scf.RHF(mol); hf.verbose = 0; hf.run()
    return hf, time.perf_counter() - t0


def _run_casci_fci(hf, norb, nelec, timeout_s=300) -> tuple:
    def _handler(signum, frame): raise TimeoutError
    signal.signal(signal.SIGALRM, _handler); signal.alarm(timeout_s)
    try:
        t0 = time.perf_counter()
        cas = mcscf.CASCI(hf, norb, nelec); cas.verbose = 0
        return cas.kernel()[0], time.perf_counter() - t0
    except TimeoutError:
        return None, None
    finally:
        signal.alarm(0)


def _run_maestro(hf, norb, nelec, ansatz, backend, mps_bond_dim=64, **kwargs) -> dict:
    n_qubits   = 2 * norb
    simulation = "statevector" if n_qubits <= SV_QUBIT_LIMIT else "mps"
    cas = mcscf.CASCI(hf, norb, nelec); cas.verbose = 0
    kw  = dict(ansatz=ansatz, backend=backend, simulation=simulation,
               verbose=False, **kwargs)
    if simulation == "mps":
        kw["mps_bond_dim"] = mps_bond_dim
    cas.fcisolver = MaestroSolver(**kw)
    try:
        t0     = time.perf_counter()
        energy = cas.kernel()[0]
        return {"status": "ok", "energy": energy,
                "time": time.perf_counter() - t0,
                "simulation": simulation,
                "mps_bond_dim": mps_bond_dim if simulation == "mps" else None,
                "converged": cas.fcisolver.converged,
                "iters": len(cas.fcisolver.energy_history)}
    except Exception as exc:
        return {"status": "failed", "error": str(exc),
                "traceback": traceback.format_exc()}


# ── Qrunch single-frame runner ────────────────────────────────────────────────

def _try_import_qrunch():
    """Import qrunch and register the license.  Returns (qc, None) or (None, reason)."""
    try:
        import qrunch as qc
        qc.register_license_file(QRUNCH_LICENSE)
        return qc, None
    except ImportError:
        return None, "qrunch not installed"
    except Exception as exc:
        return None, str(exc)


def _run_qrunch_frame(
    full_frame_atoms: list[tuple[str, float, float, float]],
    embedded_atoms: list[int],
    frame_idx: int,
    qc,
) -> dict:
    """Run Qrunch CI + FAST-VQE for a single reaction frame.

    Uses ground_state().projective_embedding() so each frame is independent
    and per-frame timing is meaningful.  Embedding results are cached in
    QRUNCH_CACHE/frame_N/ so the expensive DFT+MP2 setup is paid only once.

    Returns {"status": "ok", "e_initial", "e_ci", "t_ci", "e_vqe", "t_vqe"}
    or       {"status": "failed", "error": ..., "traceback": ...}
    """
    try:
        persister_dir = QRUNCH_CACHE / f"frame_{frame_idx}"
        persister_dir.mkdir(parents=True, exist_ok=True)

        # Write the full 27-atom frame to a temp XYZ so qrunch can read it
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False,
                                         mode="w") as tmp:
            tmp_path = Path(tmp.name)
        _write_xyz(full_frame_atoms, tmp_path, comment=f"Frame {frame_idx}")

        try:
            mol_config = qc.build_molecular_configuration(
                molecule=tmp_path,
                basis_set="sto3g",
                charge=-1,
                spin_difference=0,
                embedded_atoms=embedded_atoms,
            )

            problem_builder_creator = (
                qc.problem_builder_creator()
                .ground_state()
                .projective_embedding()
                .choose_full_system_solver().dft()
                .choose_embedded_orbital_calculator().moller_plesset_2()
                .choose_localizer().pipek_mezey()
                .choose_orbital_assigner().total_weight(assignment_tolerance=0.2)
                .choose_projector_builder().manby()
                .add_problem_modifier().active_space(
                    number_of_active_spatial_orbitals=10,
                    number_of_active_alpha_electrons=5,
                )
                .choose_data_persister_manager().file_persister(
                    directory=persister_dir, extension=".qdk",
                    load_policy="fallback",
                )
            )
            problem_builder = problem_builder_creator.create()
            problem = problem_builder.build_restricted(mol_config)
        finally:
            tmp_path.unlink(missing_ok=True)

        # CI reference
        ci_calc = (
            qc.calculator_creator()
            .configuration_interaction().standard().create()
        )
        t0       = time.perf_counter()
        ci_res   = ci_calc.calculate(problem)
        t_ci     = time.perf_counter() - t0
        e_initial = float(ci_res.initial_total_energies.values[0])
        e_ci      = float(ci_res.total_energies.values[0])

        # FAST-VQE
        estimator     = qc.estimator_creator().excitation_gate().create()
        sampler       = qc.sampler_creator().excitation_gate().create()
        gate_selector = (
            qc.gate_selector_creator()
            .fast().with_sampler(sampler).with_shots(None).create()
        )
        vqe_calc = (
            qc.calculator_creator()
            .vqe().iterative().standard()
            .choose_minimizer().last_variable_fft()
            .with_estimator(estimator)
            .with_gate_selector(gate_selector)
            .create()
        )
        t0      = time.perf_counter()
        vqe_res = vqe_calc.calculate(problem)
        t_vqe   = time.perf_counter() - t0
        e_vqe   = float(vqe_res.total_energies.values[0])

        return {"status": "ok",
                "e_initial": e_initial,
                "e_ci": e_ci,   "t_ci": t_ci,
                "e_vqe": e_vqe, "t_vqe": t_vqe}

    except Exception as exc:
        return {"status": "failed", "error": str(exc),
                "traceback": traceback.format_exc()}


# ── Formatting ─────────────────────────────────────────────────────────────────

def _fe(e):
    return f"{e:+13.6f}" if e is not None else "          N/A"

def _ft(t):
    return f"{t:6.1f}s" if t is not None else "    N/A"

def _fd(e, ref):
    if e is None or ref is None:
        return "          N/A"
    return f"{e - ref:+13.6f}"


# ── Main benchmark ─────────────────────────────────────────────────────────────

def bench_dehalogenase(
    gpu: bool = False,
    frame_indices: list[int] | None = None,
    norb: int = NORB,
    nelec: tuple[int, int] = NELEC,
    mps_bond_dim: int = 64,
    ansatz: str = "upccd",
    maxiter: int = 50,
    run_qrunch: bool = True,
) -> dict:
    """Dehalogenase SN2 — FCI / Maestro / Qrunch CI / Qrunch VQE per frame."""
    if not DEHALOGENASE_XYZ.exists():
        raise FileNotFoundError(
            f"Geometry not found: {DEHALOGENASE_XYZ}\n"
            "Copy dehalogenase_reaction_small.xyz into "
            "benchmarks/geometries/dehalogenase_data/."
        )

    # Parse all frames: full 27-atom (for Qrunch) and 5-atom subset (for PySCF)
    all_frames_full = _parse_xyz_frames(DEHALOGENASE_XYZ)
    all_frames_sub  = _parse_xyz_frames(DEHALOGENASE_XYZ, EMBEDDED_ATOM_INDICES)

    if frame_indices is None:
        frame_indices = DEFAULT_FRAMES
    n_qubits = 2 * norb

    # Check qrunch availability once
    qc, qrunch_skip_reason = (None, "--no-qrunch") if not run_qrunch \
        else _try_import_qrunch()
    qrunch_available = qc is not None

    print(f"\nDehalogenase SN2  CAS({sum(nelec)}e,{norb}o) = {n_qubits}q  "
          f"ansatz={ansatz}  χ={mps_bond_dim}  maxiter={maxiter}")
    print(f"  Frames  : {frame_indices}")
    print(f"  Qrunch  : {'available' if qrunch_available else f'skipped ({qrunch_skip_reason})'}")
    print(f"  err_M   = E_Maestro  − E_FCI        (bare PySCF variational error)")
    print(f"  err_VQE = E_FAST-VQE − E_Qrunch_CI  (Qrunch VQE variational error)")
    print(f"  ΔE      = E(frame)   − E(frame 0)   (reaction energy profile)")
    qr_note = "" if qrunch_available else "  [Qrunch cols N/A]"
    print(
        f"\n  {'Frame':>5}  "
        f"{'HF (Ha)':>13} {'t':>6}  "
        f"{'FCI (Ha)':>13} {'t':>6}  "
        f"{'Maestro (Ha)':>13} {'t':>7}  "
        f"{'err_M (Ha)':>13}  "
        f"{'Qrunch CI (Ha)':>14} {'t':>7}  "
        f"{'FAST-VQE (Ha)':>13} {'t':>7}  "
        f"{'err_VQE (Ha)':>13}"
        + qr_note
    )

    records    = []
    e_fci0     = None
    e_m0       = None
    e_qci0     = None
    e_qvqe0    = None

    for frame_idx in frame_indices:
        if frame_idx >= len(all_frames_full):
            continue

        sub_atoms  = all_frames_sub[frame_idx]
        full_atoms = all_frames_full[frame_idx]

        # ── PySCF (bare 5-atom subsystem) ─────────────────────────────────────
        mol = _mol_from_atoms(sub_atoms, charge=-1, spin=0, basis="sto-3g")
        hf, t_hf = _run_hf(mol)

        print(f"  {frame_idx:5d}  running FCI...", end="", flush=True)
        e_fci, t_fci = _run_casci_fci(hf, norb, nelec)

        print(f"  Maestro...", end="", flush=True)
        mr  = _run_maestro(hf, norb, nelec, ansatz,
                           "gpu" if gpu else "cpu",
                           mps_bond_dim=mps_bond_dim, maxiter=maxiter)
        e_m = mr.get("energy")
        t_m = mr.get("time")

        # ── Qrunch (full 27-atom system with projective embedding) ─────────────
        if qrunch_available:
            print(f"  Qrunch CI+VQE...", end="", flush=True)
            qr = _run_qrunch_frame(full_atoms, EMBEDDED_ATOM_INDICES, frame_idx, qc)
        else:
            qr = {"status": "skipped"}

        e_qci   = qr.get("e_ci")
        t_qci   = qr.get("t_ci")
        e_qvqe  = qr.get("e_vqe")
        t_qvqe  = qr.get("t_vqe")

        # Frame-0 references
        if e_fci0  is None and e_fci  is not None: e_fci0  = e_fci
        if e_m0    is None and e_m    is not None: e_m0    = e_m
        if e_qci0  is None and e_qci  is not None: e_qci0  = e_qci
        if e_qvqe0 is None and e_qvqe is not None: e_qvqe0 = e_qvqe

        err_m   = (e_m    - e_fci)  if (e_m   is not None and e_fci  is not None) else None
        err_vqe = (e_qvqe - e_qci)  if (e_qvqe is not None and e_qci is not None) else None

        print(
            f"\r  {frame_idx:5d}  "
            f"{_fe(hf.e_tot)} {_ft(t_hf)}  "
            f"{_fe(e_fci)} {_ft(t_fci)}  "
            f"{_fe(e_m)} {_ft(t_m)}  "
            f"{_fe(err_m)}  "
            f"{_fe(e_qci)} {_ft(t_qci)}  "
            f"{_fe(e_qvqe)} {_ft(t_qvqe)}  "
            f"{_fe(err_vqe)}"
        )

        records.append({
            "frame":          frame_idx,
            "e_hf":           hf.e_tot,  "t_hf":   t_hf,
            "e_fci":          e_fci,     "t_fci":  t_fci,
            "maestro":        mr,
            "err_maestro_ha": err_m,
            "delta_e_fci_ha":     (e_fci  - e_fci0)  if (e_fci  and e_fci0)  else None,
            "delta_e_maestro_ha": (e_m    - e_m0)    if (e_m    and e_m0)    else None,
            "qrunch":         qr,
            "err_qrunch_vqe_ha":  err_vqe,
            "delta_e_qci_ha":     (e_qci  - e_qci0)  if (e_qci  and e_qci0)  else None,
            "delta_e_qvqe_ha":    (e_qvqe - e_qvqe0) if (e_qvqe and e_qvqe0) else None,
        })

    # ── ΔE reaction profile summary ───────────────────────────────────────────
    if len(records) > 1:
        print(f"\n  ΔE reaction profile (Ha, relative to frame {frame_indices[0]}):")
        print(f"  {'Frame':>5}  {'ΔE_FCI':>13}  {'ΔE_Maestro':>13}  "
              f"{'ΔE_Qrunch_CI':>13}  {'ΔE_FAST-VQE':>13}")
        for r in records:
            print(
                f"  {r['frame']:5d}  "
                f"{_fd(r['e_fci'],  e_fci0):>13}  "
                f"{_fd(r['maestro'].get('energy'), e_m0):>13}  "
                f"{_fd(r['qrunch'].get('e_ci'),  e_qci0):>13}  "
                f"{_fd(r['qrunch'].get('e_vqe'), e_qvqe0):>13}"
            )

    return {
        "name":           "dehalogenase",
        "basis":          "sto-3g",
        "charge":         -1,
        "norb":           norb,
        "nelec":          list(nelec),
        "n_qubits":       n_qubits,
        "ansatz":         ansatz,
        "mps_bond_dim":   mps_bond_dim,
        "frame_indices":  frame_indices,
        "qrunch_available": qrunch_available,
        "records":        records,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dehalogenase SN2 — FCI / Maestro / Qrunch CI / Qrunch VQE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  poetry run python benchmarks/bench_dehalogenase.py
  poetry run python benchmarks/bench_dehalogenase.py --frames all
  poetry run python benchmarks/bench_dehalogenase.py --chi 32
  poetry run python benchmarks/bench_dehalogenase.py --no-qrunch
  poetry run python benchmarks/bench_dehalogenase.py --gpu
        """,
    )
    parser.add_argument("--gpu",       action="store_true")
    parser.add_argument("--chi",       type=int, default=64,
                        help="MPS bond dimension (default: 64)")
    parser.add_argument("--frames",    type=str, default=None,
                        help="Comma-separated frame indices or 'all' (default: 0,5,10)")
    parser.add_argument("--ansatz",    type=str, default="upccd",
                        choices=["upccd", "hardware_efficient"])
    parser.add_argument("--maxiter",   type=int, default=50)
    parser.add_argument("--no-qrunch", action="store_true",
                        help="Skip Qrunch (Pipeline B)")
    parser.add_argument("--output",    type=str, default=None)
    args = parser.parse_args()

    if args.frames is None:
        frame_indices = None
    elif args.frames.strip().lower() == "all":
        frame_indices = list(range(11))
    else:
        frame_indices = [int(x) for x in args.frames.split(",")]

    print("=" * 72)
    print("  DEHALOGENASE SN2 BENCHMARK")
    print(f"  GPU     : {'enabled' if args.gpu else 'disabled'}")
    print(f"  χ       : {args.chi}  |  ansatz : {args.ansatz}  |  maxiter : {args.maxiter}")
    print(f"  Frames  : {frame_indices or DEFAULT_FRAMES}")
    print(f"  Qrunch  : {'disabled' if args.no_qrunch else 'enabled (skipped if not installed)'}")
    print(f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    t0 = time.perf_counter()
    result = bench_dehalogenase(
        gpu=args.gpu,
        frame_indices=frame_indices,
        mps_bond_dim=args.chi,
        ansatz=args.ansatz,
        maxiter=args.maxiter,
        run_qrunch=not args.no_qrunch,
    )
    total_time = time.perf_counter() - t0

    output = {
        "meta": {
            "timestamp":      datetime.now().isoformat(),
            "gpu":            args.gpu,
            "python_version": platform.python_version(),
            "platform":       platform.platform(),
            "total_time_s":   round(total_time, 2),
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
    print(f"  Done in {total_time:.1f}s  |  Results: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
