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

Benchmarks the dehalogenase enzyme SN2 reaction from the Qrunch tutorial
(qrunch_tutorials/dehalogenase-tutorial) using two independent pipelines:

  Pipeline A — Bare PySCF (no embedding)
  ----------------------------------------
  Extracts the 5 embedded reaction-center atoms (C, O, O, Cl, C) directly
  into PySCF.  Same CAS(10e,10o)/STO-3G active space as the tutorial, but
  without environmental embedding.  Always runnable.

    Methods:  HF  |  CASCI/FCI (exact reference)  |  Maestro UpCCD (MPS)

  Pipeline B — Qrunch projective embedding
  -----------------------------------------
  Replicates the tutorial exactly: DFT mean field for the 27-atom environment,
  MP2 natural orbitals, Pipek-Mezey localisation, Manby level-shift projector.
  Embedding results are cached in benchmarks/cache/dehalogenase_qrunch/ so the
  expensive setup (~25 min first run) is paid only once.
  Requires the qrunch package (Linux only; skipped gracefully on Mac/without it).

    Methods:  initial (HF-like)  |  Qrunch CI (exact)  |  Qrunch FAST-VQE

In both pipelines:
  err  = E_VQE - E_exact  (variational error of the quantum solver)
  ΔE   = E(frame) - E(frame 0)  (reaction energy profile / barrier height)

Runtime notes
-------------
  Pipeline A — each frame: MPS VQE (20q, χ=64, 50 iters) ≈ 5–15 min CPU.
    3 frames  (default: 0, 5, 10)  :  ~15–45 min
    11 frames (--frames all)       :  ~1–3 hours
  Pipeline B — first run: ~25 min embedding setup (cached afterwards).
    CI + FAST-VQE across 11 frames: ~30–90 min after cache is warm.

Usage
-----
    poetry run python benchmarks/bench_dehalogenase.py              # frames 0,5,10
    poetry run python benchmarks/bench_dehalogenase.py --frames all # all 11 frames
    poetry run python benchmarks/bench_dehalogenase.py --chi 32     # faster MPS
    poetry run python benchmarks/bench_dehalogenase.py --gpu        # GPU backend
    poetry run python benchmarks/bench_dehalogenase.py --no-qrunch  # skip Pipeline B
"""

import argparse
import json
import platform
import signal
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

# Full 27-atom reaction XYZ (used by both pipelines)
DEHALOGENASE_XYZ = (
    ROOT / "benchmarks/geometries/dehalogenase_data/dehalogenase_reaction_small.xyz"
)
# 0-based indices of the 5 embedded reaction-center atoms: C, O, O, Cl, C
EMBEDDED_ATOM_INDICES = [5, 6, 7, 18, 19]

# Active space matching the Qrunch tutorial
NORB  = 10      # 10 spatial orbitals → 20 qubits
NELEC = (5, 5)  # 10 electrons, restricted (5α + 5β)

SV_QUBIT_LIMIT = 14  # 20q > limit → MPS


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
    """Parse a multi-frame XYZ file into a list of atom lists.

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
        i += 2  # skip n_atoms line and frame-label comment
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rel_energy_ha(e, e_ref):
    if e is None or e_ref is None:
        return None
    return e - e_ref


# ── Pipeline A: bare PySCF runners ────────────────────────────────────────────

def _run_hf(mol: gto.Mole) -> tuple[scf.hf.SCF, float]:
    t0 = time.perf_counter()
    hf = scf.RHF(mol)
    hf.verbose = 0
    hf.run()
    return hf, time.perf_counter() - t0


def _run_casci_fci(hf: scf.hf.SCF, norb: int, nelec: tuple[int, int],
                   timeout_s: int = 300) -> tuple[float | None, float | None]:
    """CASCI with PySCF FCI — exact in the active space."""
    def _handler(signum, frame): raise TimeoutError
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
        return {"status": "failed", "error": str(exc),
                "traceback": traceback.format_exc()}


# ── Pipeline B: qrunch embedding ──────────────────────────────────────────────

def _run_qrunch_pipeline(
    xyz_path: Path,
    embedded_atoms: list[int],
    persister_dir: Path,
    run_vqe: bool = True,
) -> dict:
    """Run the full Qrunch projective-embedding pipeline.

    Mirrors the tutorial exactly:
      DFT full system → MP2 embedded orbitals → Pipek-Mezey localisation
      → total-weight orbital assignment → Manby projector
      → active space CAS(10e, 10o) → CI + FAST-VQE

    Embedding results are cached in persister_dir so the expensive setup
    (~25 min first run) is only paid once.

    Returns a dict with status "ok" and per-frame results, or status "skipped"
    when qrunch is not installed/compatible.
    """
    try:
        import qrunch as qc
    except ImportError:
        return {"status": "skipped", "reason": "qrunch not installed"}
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}

    try:
        persister_dir.mkdir(exist_ok=True, parents=True)

        # ── Build reaction configuration ──────────────────────────────────────
        reaction = qc.build_reaction_configuration(
            reaction=xyz_path,
            basis_set="sto3g",
            charge=-1,
            spin_difference=0,
            embedded_atoms=embedded_atoms,
        )

        # ── Build problem (embedding setup) ───────────────────────────────────
        reaction_builder_creator = (
            qc.problem_builder_creator()
            .reaction_path()
            .even_handed()
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
                directory=persister_dir, extension=".qdk", load_policy="fallback"
            )
        )
        reaction_builder = reaction_builder_creator.create()

        print(f"  [Qrunch] Building embedded reaction problem "
              f"(cached in {persister_dir}) ...", flush=True)
        t0 = time.perf_counter()
        reaction_problem = reaction_builder.build_restricted(reaction)
        t_setup = time.perf_counter() - t0
        print(f"  [Qrunch] Setup done in {t_setup:.1f}s")

        # ── CI reference (exact in active space) ──────────────────────────────
        ci_calculator = (
            qc.calculator_creator()
            .configuration_interaction()
            .standard()
            .create()
        )
        print(f"  [Qrunch] Running CI ...", flush=True)
        t0 = time.perf_counter()
        ci_result = ci_calculator.calculate(reaction_problem)
        t_ci = time.perf_counter() - t0
        print(f"  [Qrunch] CI done in {t_ci:.1f}s")

        # ── FAST-VQE ──────────────────────────────────────────────────────────
        vqe_result = None
        t_vqe = None
        if run_vqe:
            estimator = qc.estimator_creator().excitation_gate().create()
            sampler   = qc.sampler_creator().excitation_gate().create()
            gate_selector = (
                qc.gate_selector_creator()
                .fast()
                .with_sampler(sampler)
                .with_shots(None)
                .create()
            )
            vqe_calculator = (
                qc.calculator_creator()
                .vqe()
                .iterative()
                .standard()
                .choose_minimizer().last_variable_fft()
                .with_estimator(estimator)
                .with_gate_selector(gate_selector)
                .create()
            )
            print(f"  [Qrunch] Running FAST-VQE ...", flush=True)
            t0 = time.perf_counter()
            vqe_result = vqe_calculator.calculate(reaction_problem)
            t_vqe = time.perf_counter() - t0
            print(f"  [Qrunch] FAST-VQE done in {t_vqe:.1f}s")

        # ── Extract per-frame results ──────────────────────────────────────────
        ci_energies      = list(ci_result.total_energies.values)
        initial_energies = list(ci_result.initial_total_energies.values)
        vqe_energies     = (list(vqe_result.total_energies.values)
                            if vqe_result is not None else [None] * len(ci_energies))

        frames = {}
        for i, (e_init, e_ci, e_vqe) in enumerate(
                zip(initial_energies, ci_energies, vqe_energies)):
            frames[i] = {
                "e_initial": float(e_init),
                "e_ci":  float(e_ci),
                "e_vqe": float(e_vqe) if e_vqe is not None else None,
            }

        return {
            "status": "ok",
            "t_setup": t_setup,
            "t_ci":    t_ci,
            "t_vqe":   t_vqe,
            "n_frames": len(frames),
            "frames": frames,
        }

    except Exception as exc:
        return {"status": "failed", "error": str(exc),
                "traceback": traceback.format_exc()}


# ── Print helpers ──────────────────────────────────────────────────────────────

def _s(e):
    return f"{e:+13.6f}" if e is not None else "          N/A"

def _t(t):
    return f"{t:5.1f}s" if t is not None else "   N/A"

def _sd(e, ref):
    v = _rel_energy_ha(e, ref)
    return f"{v:+12.6f}" if v is not None else "         N/A "


def _print_section(title: str, frame_rows: list[dict]) -> None:
    """Print a formatted per-frame energy table.

    Each row dict must have keys:
      frame, e_ref, t_ref, e_exact, t_exact, e_vqe, t_vqe,
      label_ref, label_exact, label_vqe
    """
    print(f"\n  {title}")
    print(f"  err = E_VQE - E_exact  |  ΔE = E(frame) - E(frame 0)")
    row0 = frame_rows[0]
    print(
        f"  {'Frame':>5}  "
        f"{row0['label_ref']:>13} {'t':>5}  "
        f"{row0['label_exact']:>13} {'t':>5}  "
        f"{row0['label_vqe']:>13} {'t':>7}  "
        f"{'err (Ha)':>13}  "
        f"{'ΔE_exact (Ha)':>13}  "
        f"{'ΔE_VQE (Ha)':>12}"
    )

    e_exact0 = None
    e_vqe0   = None
    for row in frame_rows:
        e_exact = row["e_exact"]
        e_vqe   = row["e_vqe"]
        if e_exact0 is None and e_exact is not None:
            e_exact0 = e_exact
        if e_vqe0 is None and e_vqe is not None:
            e_vqe0 = e_vqe
        err = (e_vqe - e_exact) if (e_vqe is not None and e_exact is not None) else None
        print(
            f"  {row['frame']:5d}  "
            f"{_s(row['e_ref'])} {_t(row['t_ref'])}  "
            f"{_s(e_exact)} {_t(row['t_exact'])}  "
            f"{_s(e_vqe)} {_t(row['t_vqe']):>7}  "
            f"{_s(err)}  "
            f"{_sd(e_exact, e_exact0):>13}  "
            f"{_sd(e_vqe, e_vqe0):>12}"
        )


# ── Main benchmark ─────────────────────────────────────────────────────────────

DEFAULT_FRAMES = [0, 5, 10]  # reactant, ~TS, product

QRUNCH_PERSISTER_DIR = CACHE_DIR / "dehalogenase_qrunch"


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
    """Dehalogenase SN2 reaction — two-pipeline benchmark.

    Pipeline A: bare PySCF (HF / CASCI-FCI / Maestro UpCCD MPS)
    Pipeline B: Qrunch projective embedding (initial / CI / FAST-VQE)
                — skipped gracefully if qrunch is not installed.
    """
    if not DEHALOGENASE_XYZ.exists():
        raise FileNotFoundError(
            f"Geometry file not found: {DEHALOGENASE_XYZ}\n"
            "Copy dehalogenase_reaction_small.xyz into "
            "benchmarks/geometries/dehalogenase_data/."
        )

    all_frames = _parse_xyz_frames(DEHALOGENASE_XYZ, EMBEDDED_ATOM_INDICES)
    if frame_indices is None:
        frame_indices = DEFAULT_FRAMES

    selected = [(i, all_frames[i]) for i in frame_indices if i < len(all_frames)]
    n_qubits = 2 * norb

    print(f"\nDehalogenase SN2  CAS({sum(nelec)}e,{norb}o) = {n_qubits}q")
    print(f"  Frames : {frame_indices}")

    # ── Pipeline A: bare PySCF ────────────────────────────────────────────────
    print(f"\n  Pipeline A — bare PySCF (5-atom subsystem, STO-3G, charge=-1)")
    print(f"    Maestro: {ansatz}  MPS χ={mps_bond_dim}  maxiter={maxiter}")

    pyscf_records = []
    for frame_idx, atoms in selected:
        mol = _mol_from_atoms(atoms, charge=-1, spin=0, basis="sto-3g")
        hf, t_hf = _run_hf(mol)

        print(f"  {frame_idx:5d}  running FCI...", end="", flush=True)
        e_fci, t_fci = _run_casci_fci(hf, norb, nelec)

        print(f"  running Maestro...", end="", flush=True)
        m = _run_maestro(hf, norb, nelec, ansatz,
                         "gpu" if gpu else "cpu",
                         mps_bond_dim=mps_bond_dim, maxiter=maxiter)
        e_m = m.get("energy")

        print(f"\r  frame {frame_idx} done", flush=True)

        pyscf_records.append({
            "frame":          frame_idx,
            "e_hf":           hf.e_tot, "t_hf": t_hf,
            "e_fci":          e_fci,    "t_fci": t_fci,
            "maestro":        m,
            "err_maestro_ha": (e_m - e_fci) if (e_m is not None and e_fci is not None) else None,
        })

    # Print Pipeline A table
    e_fci0 = next((r["e_fci"] for r in pyscf_records if r["e_fci"] is not None), None)
    e_m0   = next((r["maestro"].get("energy") for r in pyscf_records
                   if r["maestro"].get("energy") is not None), None)
    rows_a = []
    for r in pyscf_records:
        e_m = r["maestro"].get("energy")
        rows_a.append({
            "frame":       r["frame"],
            "label_ref":   "HF (Ha)",
            "label_exact": "FCI (Ha)",
            "label_vqe":   "Maestro (Ha)",
            "e_ref":       r["e_hf"],  "t_ref":   r["t_hf"],
            "e_exact":     r["e_fci"], "t_exact": r["t_fci"],
            "e_vqe":       e_m,        "t_vqe":   r["maestro"].get("time"),
        })
    _print_section("Pipeline A — bare PySCF", rows_a)

    # Add ΔE to records after we know frame-0 reference
    for r, row in zip(pyscf_records, rows_a):
        e_m = r["maestro"].get("energy")
        r["delta_e_fci_ha"]     = _rel_energy_ha(r["e_fci"], e_fci0)
        r["delta_e_maestro_ha"] = _rel_energy_ha(e_m, e_m0)

    # ── Pipeline B: Qrunch embedding ──────────────────────────────────────────
    qrunch_result = {"status": "skipped", "reason": "--no-qrunch flag"}
    if run_qrunch:
        print(f"\n  Pipeline B — Qrunch projective embedding (27-atom full system)")
        qrunch_result = _run_qrunch_pipeline(
            xyz_path=DEHALOGENASE_XYZ,
            embedded_atoms=EMBEDDED_ATOM_INDICES,
            persister_dir=QRUNCH_PERSISTER_DIR,
        )

    if qrunch_result["status"] == "ok":
        qf = qrunch_result["frames"]
        # Select only requested frames
        rows_b = []
        for fi in frame_indices:
            if fi not in qf:
                continue
            fd = qf[fi]
            rows_b.append({
                "frame":       fi,
                "label_ref":   "Initial (Ha)",
                "label_exact": "Qrunch CI (Ha)",
                "label_vqe":   "FAST-VQE (Ha)",
                "e_ref":   fd["e_initial"], "t_ref":   None,
                "e_exact": fd["e_ci"],      "t_exact": None,
                "e_vqe":   fd["e_vqe"],     "t_vqe":   None,
            })
        if rows_b:
            _print_section("Pipeline B — Qrunch embedding", rows_b)

        # Add ΔE to qrunch frame records
        e_ci0  = qf.get(frame_indices[0], {}).get("e_ci")
        e_vqe0 = qf.get(frame_indices[0], {}).get("e_vqe")
        for fi in frame_indices:
            if fi in qf:
                fd = qf[fi]
                fd["delta_e_ci_ha"]  = _rel_energy_ha(fd["e_ci"],  e_ci0)
                fd["delta_e_vqe_ha"] = _rel_energy_ha(fd["e_vqe"], e_vqe0)

    elif run_qrunch:
        reason = qrunch_result.get("reason", qrunch_result.get("error", "unknown"))
        print(f"\n  Pipeline B — skipped: {reason}")

    return {
        "name":        "dehalogenase",
        "basis":       "sto-3g",
        "charge":      -1,
        "norb":        norb,
        "nelec":       list(nelec),
        "n_qubits":    n_qubits,
        "ansatz":      ansatz,
        "mps_bond_dim": mps_bond_dim,
        "frame_indices": frame_indices,
        "pipeline_a":  pyscf_records,
        "pipeline_b":  qrunch_result,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dehalogenase SN2 benchmark — bare PySCF + Qrunch embedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  poetry run python benchmarks/bench_dehalogenase.py                 # frames 0,5,10
  poetry run python benchmarks/bench_dehalogenase.py --frames all    # all 11 frames
  poetry run python benchmarks/bench_dehalogenase.py --chi 32        # faster MPS
  poetry run python benchmarks/bench_dehalogenase.py --no-qrunch     # skip Pipeline B
  poetry run python benchmarks/bench_dehalogenase.py --gpu
        """,
    )
    parser.add_argument("--gpu",       action="store_true", help="Maestro GPU backend")
    parser.add_argument("--chi",       type=int, default=64,
                        help="MPS bond dimension χ (default: 64)")
    parser.add_argument("--frames",    type=str, default=None,
                        help="Comma-separated frame indices or 'all' (default: 0,5,10)")
    parser.add_argument("--ansatz",    type=str, default="upccd",
                        choices=["upccd", "hardware_efficient"],
                        help="Maestro VQE ansatz (default: upccd)")
    parser.add_argument("--maxiter",   type=int, default=50,
                        help="Max Maestro VQE iterations per frame (default: 50)")
    parser.add_argument("--no-qrunch", action="store_true",
                        help="Skip Pipeline B (Qrunch embedding)")
    parser.add_argument("--output",    type=str, default=None,
                        help="Override output JSON path")
    args = parser.parse_args()

    if args.frames is None:
        frame_indices = None
    elif args.frames.strip().lower() == "all":
        frame_indices = list(range(11))
    else:
        frame_indices = [int(x) for x in args.frames.split(",")]

    print("=" * 72)
    print("  DEHALOGENASE SN2 BENCHMARK")
    print(f"  GPU       : {'enabled' if args.gpu else 'disabled'}")
    print(f"  χ         : {args.chi}")
    print(f"  Ansatz    : {args.ansatz}")
    print(f"  Frames    : {frame_indices or DEFAULT_FRAMES}")
    print(f"  Qrunch    : {'disabled (--no-qrunch)' if args.no_qrunch else 'enabled (skipped if not installed)'}")
    print(f"  Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
