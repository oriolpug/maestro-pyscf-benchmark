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
  Qoro         — UpCCD MPS VQE (bare PySCF + QoroSolver)
  Qrunch CI    — FCI on the Qrunch-embedded Hamiltonian
                 (DFT env + MP2 orbitals + Pipek-Mezey + Manby projector)
  Qrunch VQE   — FAST-VQE (adaptive excitation-gate) on the embedded Hamiltonian

Modes
-----
  Standard (default): All solvers run on the isolated 5-atom subsystem
  (27-atom geometry, STO-3G, max norb=13).  Energies are directly comparable.

  --big: Uses the 86-atom geometry with Qrunch projective embedding.
  Qrunch builds the embedded Hamiltonian (DFT env + MP2 orbitals + Manby
  projector), then ALL solvers (FCI, QSCI, Qoro, FAST-VQE) run on that
  same active-space Hamiltonian.  Defaults: norb=20 (40 qubits), basis=pc-seg-1.
  Use --norb to scale the active space up or down.

  err_M   = E_Qoro     − E_FCI       (Qoro variational error)
  err_VQE = E_FAST-VQE − E_FCI       (FAST-VQE variational error)
  ΔE      = E(frame)   − E(frame 0)  (reaction energy profile)

Runtime notes
-------------
  Qoro MPS (20q, χ=64, 50 iters) ≈ 5–15 min/frame CPU.
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
import os
import platform
import shutil
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

from qoro_pyscf import QoroSolver, QSCISolver

CACHE_DIR    = Path(__file__).parent / "cache"
QRUNCH_LICENSE = ROOT / "benchmarks/qrunch/license.txt"
QRUNCH_CACHE   = CACHE_DIR / "dehalogenase_qrunch"

DEHALOGENASE_SMALL_XYZ = (
    ROOT / "benchmarks/geometries/dehalogenase_data/dehalogenase_reaction_small.xyz"
)
DEHALOGENASE_LARGE_XYZ = (
    ROOT / "benchmarks/geometries/dehalogenase_data/dehalogenase_reaction_large.xyz"
)
EMBEDDED_ATOM_INDICES_SMALL = [5, 6, 7, 18, 19]  # C, O, O, Cl, C (0-based, 27-atom)
EMBEDDED_ATOM_INDICES_LARGE = [5, 6, 7, 74, 75]  # C, O, O, Cl, C (0-based, 86-atom)

NORB  = 10
NELEC = (5, 5)
BIG_NORB  = 20   # 40 qubits, matching Kvantify showcase
BIG_BASIS = "pc-seg-1"
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


def _run_vqe_qoro(hf, norb, nelec, ansatz, backend, mps_bond_dim=64, **kwargs) -> dict:
    n_qubits   = 2 * norb
    simulation = "statevector" if n_qubits <= SV_QUBIT_LIMIT else "mps"
    cas = mcscf.CASCI(hf, norb, nelec); cas.verbose = 0
    kw  = dict(ansatz=ansatz, backend=backend, simulation=simulation,
               verbose=False, **kwargs)
    if simulation == "mps":
        kw["mps_bond_dim"] = mps_bond_dim
    cas.fcisolver = QoroSolver(**kw)
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


# ── QSCI runner (qiskit-addon-sqd) ───────────────────────────────────────────

def _run_qsci(hf, norb, nelec, num_samples: int = 200, rand_seed: int = 42) -> dict:
    """QSCI via qiskit-addon-sqd with uniform random bitstring sampling.

    Projects the Hamiltonian onto the subspace spanned by `num_samples`
    randomly sampled bitstrings with the correct electron count, then
    diagonalises exactly within that subspace.  Uses the same PySCF
    active-space integrals as FCI and Qoro — all three methods are
    directly comparable.

    In production QSCI the bitstrings come from sampling a quantum circuit
    (e.g. a hardware VQE state); here uniform random sampling is used as a
    tractable classical proxy.  At n_samples ≈ 1000 the random subspace
    covers enough of the FCI space to recover the exact energy for this
    active space.
    """
    try:
        from qiskit_addon_sqd.fermion import solve_fermion
        from qiskit_addon_sqd.counts import (
            generate_counts_bipartite_hamming, counts_to_arrays,
        )
        from pyscf import ao2mo
    except ImportError as exc:
        return {"status": "failed", "error": str(exc)}

    try:
        nalpha, nbeta = nelec
        cas = mcscf.CASCI(hf, norb, nelec)
        cas.verbose = 0
        h1e, ecore = cas.get_h1eff()
        h2e = ao2mo.restore(1, cas.get_h2eff(), norb)

        counts = generate_counts_bipartite_hamming(
            num_samples, 2 * norb,
            hamming_right=nalpha,  # alpha (spin-up) in right half
            hamming_left=nbeta,    # beta (spin-down) in left half
            rand_seed=rand_seed,
        )
        bsm, _ = counts_to_arrays(counts)  # shape (n_unique, 2*norb), bool

        t0 = time.perf_counter()
        e_elec, _, _, _ = solve_fermion(bsm, hcore=h1e, eri=h2e)
        elapsed = time.perf_counter() - t0

        return {
            "status":     "ok",
            "energy":     float(e_elec + ecore),
            "time":       elapsed,
            "n_samples":  num_samples,
            "n_det":      int(bsm.shape[0]),  # unique determinants after dedup
        }
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n  [QSCI] ERROR: {exc}\n{tb}")
        return {"status": "failed", "error": str(exc), "traceback": tb}


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
        tb = traceback.format_exc()
        print(f"  [Qrunch] import/license error:\n{tb}")
        return None, str(exc)


def _build_standard_problem(
    sub_frame_atoms: list[tuple[str, float, float, float]],
    norb: int,
    nelec_alpha: int,
    frame_idx: int,
    qc,
):
    """Build a Qrunch standard (non-embedded) problem for the isolated subsystem."""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False,
                                     mode="w") as tmp:
        tmp_path = Path(tmp.name)
    _write_xyz(sub_frame_atoms, tmp_path, comment=f"Frame {frame_idx}")

    try:
        mol_config = qc.build_molecular_configuration(
            molecule=tmp_path,
            basis_set="sto3g",
            charge=-1,
            spin_difference=0,
        )

        problem_builder_creator = (
            qc.problem_builder_creator()
            .ground_state()
            .standard()
            .add_problem_modifier().active_space(
                number_of_active_spatial_orbitals=norb,
                number_of_active_alpha_electrons=nelec_alpha,
            )
        )
        return problem_builder_creator.create().build_restricted(mol_config)
    finally:
        tmp_path.unlink(missing_ok=True)


def _check_tmpdir_space(full_frame_atoms, basis_set):
    """Estimate temp disk needed for AO-to-MO integral transformation and
    raise early if the tmpdir is too small.

    The bottleneck is PySCF's ao2mo outcore half-transformation, which writes
    intermediate HDF5 files scaling roughly as nao^2 * n_emb * 8 bytes.
    We use a conservative empirical formula calibrated against observed usage.
    """
    tmpdir = Path(tempfile.gettempdir())
    free_bytes = shutil.disk_usage(tmpdir).free

    # Build a throwaway PySCF mol just to count AOs
    atom_str = "; ".join(f"{e} {x:.8f} {y:.8f} {z:.8f}"
                         for e, x, y, z in full_frame_atoms)
    pyscf_basis = basis_set.replace("pcseg", "pc-seg-").replace("pcsseg", "pc-sseg-")
    if "seg" in basis_set and "-" not in basis_set:
        # e.g. "pcseg1" -> "pc-seg-1"
        pyscf_basis = basis_set.replace("pcseg", "pc-seg-")
    else:
        pyscf_basis = basis_set.replace("sto3g", "sto-3g")
    try:
        mol = gto.M(atom=atom_str, basis=pyscf_basis, charge=-1, spin=0,
                     verbose=0, unit="Angstrom")
        nao = mol.nao_nr()
    except Exception:
        return  # can't estimate, skip the check

    # Empirical: ao2mo temp ≈ 7.55e-5 * nao^1.89 GB
    # Calibrated from: 274 AOs (sto-3g) ≈ 3 GB, 843 AOs (pc-seg-1) ≈ 25 GB
    estimated_gb = 7.55e-5 * (nao ** 1.89)
    free_gb = free_bytes / 1e9

    if estimated_gb > free_gb:
        raise OSError(
            f"Insufficient temp disk space for AO-to-MO integral transformation.\n"
            f"  Basis {basis_set} on {len(full_frame_atoms)} atoms → {nao} AOs\n"
            f"  Estimated temp space: ~{estimated_gb:.0f} GB\n"
            f"  Available in {tmpdir}: {free_gb:.1f} GB\n"
            f"\n"
            f"Options:\n"
            f"  1. Use a smaller basis:  --basis sto-3g  (~3 GB)\n"
            f"  2. Point to a larger disk:  TMPDIR=/path/to/large/disk poetry run ...\n"
            f"  3. Free disk space on {tmpdir.resolve()}"
        )


def _build_embedded_problem(
    full_frame_atoms: list[tuple[str, float, float, float]],
    embedded_atoms: list[int],
    norb: int,
    nelec_alpha: int,
    frame_idx: int,
    qc,
    basis_set: str = "sto3g",
):
    """Build a Qrunch projective-embedding problem for the full system.

    Returns the RestrictedGroundStateProblem with the embedded Hamiltonian.
    Embedding results are cached in QRUNCH_CACHE/frame_N/.
    """
    # Check disk space before the expensive embedding
    _check_tmpdir_space(full_frame_atoms, basis_set)

    persister_dir = QRUNCH_CACHE / f"frame_{frame_idx}"
    persister_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False,
                                     mode="w") as tmp:
        tmp_path = Path(tmp.name)
    _write_xyz(full_frame_atoms, tmp_path, comment=f"Frame {frame_idx}")

    try:
        mol_config = qc.build_molecular_configuration(
            molecule=tmp_path,
            basis_set=basis_set,
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
                number_of_active_spatial_orbitals=norb,
                number_of_active_alpha_electrons=nelec_alpha,
            )
            .choose_data_persister_manager().file_persister(
                directory=persister_dir, extension=".qdk",
                load_policy="fallback",
            )
        )
        problem = problem_builder_creator.create().build_restricted(mol_config)
    finally:
        tmp_path.unlink(missing_ok=True)

    return problem


def _extract_integrals(problem):
    """Extract h1e, h2e, ecore from a Qrunch RestrictedGroundStateProblem.

    Returns (h1e, h2e, ecore) where ecore = environment + nuclear + inactive
    energy offset, so that E_total = ecore + E_active(h1e, h2e).
    """
    esi   = problem.electronic_structure_integrals
    h1e   = np.array(esi.one_body_core_hamiltonian.alpha_alpha)
    h2e   = np.array(esi.two_body_electron_repulsion_integrals.alpha_alpha)
    ecore = sum(v for _, v in problem.energy_contributions.items())
    return h1e, h2e, float(ecore)


def _run_qrunch_ci(problem, frame_idx, qc) -> dict:
    """Run Qrunch CI (configuration interaction) on an already-built problem."""
    try:
        ci_calc = (
            qc.calculator_creator()
            .configuration_interaction().standard().create()
        )
        t0     = time.perf_counter()
        ci_res = ci_calc.calculate(problem)
        t_ci   = time.perf_counter() - t0
        e_ci   = float(ci_res.total_energy.value)
        return {"status": "ok", "energy": e_ci, "time": t_ci}
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n  [Qrunch CI frame {frame_idx}] ERROR: {exc}\n{tb}")
        return {"status": "failed", "error": str(exc), "traceback": tb}


def _run_qrunch_vqe(problem, frame_idx, qc) -> dict:
    """Run Qrunch FAST-VQE on an already-built problem."""
    try:
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
        e_vqe   = float(vqe_res.total_energy.value)
        return {"status": "ok", "energy": e_vqe, "time": t_vqe}
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n  [Qrunch VQE frame {frame_idx}] ERROR: {exc}\n{tb}")
        return {"status": "failed", "error": str(exc), "traceback": tb}


def _run_fci_on_integrals(h1e, h2e, norb, nelec, ecore, timeout_s=300):
    """Run PySCF FCI directly on extracted integrals."""
    from pyscf import fci
    def _handler(signum, frame): raise TimeoutError
    signal.signal(signal.SIGALRM, _handler); signal.alarm(timeout_s)
    try:
        t0 = time.perf_counter()
        e_act, _ = fci.direct_spin1.kernel(h1e, h2e, norb, nelec, verbose=0)
        return ecore + e_act, time.perf_counter() - t0
    except TimeoutError:
        return None, None
    finally:
        signal.alarm(0)


def _run_vqe_qoro_on_integrals(h1e, h2e, norb, nelec, ecore,
                               ansatz, backend, mps_bond_dim=64,
                               **kwargs) -> dict:
    """Run Qoro VQE directly on extracted integrals."""
    n_qubits   = 2 * norb
    simulation = "statevector" if n_qubits <= SV_QUBIT_LIMIT else "mps"
    kw = dict(ansatz=ansatz, backend=backend, simulation=simulation,
              verbose=False, **kwargs)
    if simulation == "mps":
        kw["mps_bond_dim"] = mps_bond_dim
    solver = QoroSolver(**kw)
    try:
        t0 = time.perf_counter()
        e_act, _ = solver.kernel(h1e, h2e, norb, nelec, ecore=0)
        energy = ecore + e_act
        return {"status": "ok", "energy": energy,
                "time": time.perf_counter() - t0,
                "simulation": simulation,
                "mps_bond_dim": mps_bond_dim if simulation == "mps" else None,
                "converged": solver.converged,
                "iters": len(solver.energy_history)}
    except Exception as exc:
        return {"status": "failed", "error": str(exc),
                "traceback": traceback.format_exc()}


def _run_qsci_on_integrals(h1e, h2e, norb, nelec, ecore,
                            num_samples=200, rand_seed=42) -> dict:
    """Run QSCI directly on extracted integrals."""
    try:
        from qiskit_addon_sqd.fermion import solve_fermion
        from qiskit_addon_sqd.counts import (
            generate_counts_bipartite_hamming, counts_to_arrays,
        )
    except ImportError as exc:
        return {"status": "failed", "error": str(exc)}
    try:
        nalpha, nbeta = nelec
        counts = generate_counts_bipartite_hamming(
            num_samples, 2 * norb,
            hamming_right=nalpha,
            hamming_left=nbeta,
            rand_seed=rand_seed,
        )
        bsm, _ = counts_to_arrays(counts)

        t0 = time.perf_counter()
        e_elec, _, _, _ = solve_fermion(bsm, hcore=h1e, eri=h2e)
        elapsed = time.perf_counter() - t0

        return {
            "status":    "ok",
            "energy":    float(e_elec + ecore),
            "time":      elapsed,
            "n_samples": num_samples,
            "n_det":     int(bsm.shape[0]),
        }
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n  [QSCI] ERROR: {exc}\n{tb}")
        return {"status": "failed", "error": str(exc), "traceback": tb}


# ── QSCI (qoro-pyscf QSCISolver) runners ─────────────────────────────────────

def _run_qsci_qoro(hf, norb, nelec, ansatz, backend, mps_bond_dim=64,
                   n_samples=500, **kwargs) -> dict:
    """QSCI via qoro-pyscf QSCISolver (VQE trial state + selected-CI diagonalization)."""
    n_qubits   = 2 * norb
    simulation = "statevector" if n_qubits <= SV_QUBIT_LIMIT else "mps"
    cas = mcscf.CASCI(hf, norb, nelec); cas.verbose = 0
    cas.canonicalization = False
    qoro_kw = dict(ansatz=ansatz, backend=backend, simulation=simulation,
                   verbose=False, **kwargs)
    if simulation == "mps":
        qoro_kw["mps_bond_dim"] = mps_bond_dim
    inner = QoroSolver(**qoro_kw)
    cas.fcisolver = QSCISolver(inner_solver=inner, n_samples=n_samples, verbose=False)
    try:
        t0     = time.perf_counter()
        energy = cas.kernel()[0]
        return {"status": "ok", "energy": energy,
                "time": time.perf_counter() - t0,
                "simulation": simulation, "n_samples": n_samples, "converged": True}
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n  [QSCI-Qoro] ERROR: {exc}\n{tb}")
        return {"status": "failed", "error": str(exc), "traceback": tb}


def _run_qsci_qoro_on_integrals(h1e, h2e, norb, nelec, ecore,
                                 ansatz, backend, mps_bond_dim=64,
                                 n_samples=500, **kwargs) -> dict:
    """Run QSCISolver directly on extracted integrals (big molecule / Qrunch path)."""
    n_qubits   = 2 * norb
    simulation = "statevector" if n_qubits <= SV_QUBIT_LIMIT else "mps"
    qoro_kw = dict(ansatz=ansatz, backend=backend, simulation=simulation,
                   verbose=False, **kwargs)
    if simulation == "mps":
        qoro_kw["mps_bond_dim"] = mps_bond_dim
    inner = QoroSolver(**qoro_kw)
    solver = QSCISolver(inner_solver=inner, n_samples=n_samples, verbose=False)
    try:
        t0 = time.perf_counter()
        e_act, _ = solver.kernel(h1e, h2e, norb, nelec, ecore=0)
        energy = ecore + e_act
        return {"status": "ok", "energy": energy,
                "time": time.perf_counter() - t0,
                "simulation": simulation, "n_samples": n_samples}
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n  [QSCI-Qoro] ERROR: {exc}\n{tb}")
        return {"status": "failed", "error": str(exc), "traceback": tb}


# ── Qiskit VQE runner ────────────────────────────────────────────────────────

def _run_qiskit_vqe_on_integrals(h1e, h2e, norb, nelec, ecore,
                                  mps_bond_dim=64, maxiter=200) -> dict:
    """Run Qiskit VQE (UpCCD ansatz, COBYLA optimizer) on extracted integrals.

    Uses JordanWignerMapper (2*norb qubits) with statevector simulation for
    small systems or MPS via Aer for larger ones.
    """
    try:
        from scipy.optimize import minimize as _minimize
        from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.second_q.problems import (
            ElectronicBasis, ElectronicStructureProblem)
    except ImportError as exc:
        return {"status": "failed", "error": str(exc)}

    try:
        nalpha, nbeta = nelec

        hamiltonian = ElectronicEnergy.from_raw_integrals(h1e, h2e)
        problem = ElectronicStructureProblem(hamiltonian)
        problem.basis = ElectronicBasis.MO
        problem.num_spatial_orbitals = norb
        problem.num_particles = (nalpha, nbeta)

        mapper   = JordanWignerMapper()
        qubit_op = mapper.map(problem.second_q_ops()[0])
        n_qubits = 2 * norb

        # Build UpCCD circuit (same as benchmarks.py)
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        n_occ = min(nalpha, nbeta)
        occ_spatial = list(range(n_occ))
        vir_spatial = list(range(max(nalpha, nbeta), norb))
        pairs = [(i, a) for i in occ_spatial for a in vir_spatial]
        n_params = len(pairs)
        params = ParameterVector("θ", n_params)
        ansatz = QuantumCircuit(n_qubits)

        # Hartree-Fock initial state
        for i in range(nalpha):
            ansatz.x(2 * i)
        for i in range(nbeta):
            ansatz.x(2 * i + 1)

        # Paired double excitations: 6 CNOT + 1 Ry per pair
        for idx, (i_spat, a_spat) in enumerate(pairs):
            p, q, r, s = sorted([2*i_spat, 2*i_spat+1, 2*a_spat, 2*a_spat+1])
            ansatz.cx(p, q)
            ansatz.cx(r, s)
            ansatz.cx(p, r)
            ansatz.ry(2 * params[idx], p)
            ansatz.cx(p, r)
            ansatz.cx(r, s)
            ansatz.cx(p, q)

        if n_qubits <= SV_QUBIT_LIMIT:
            from qiskit.primitives import StatevectorEstimator
            est = StatevectorEstimator()
            simulation = "statevector"
        else:
            from qiskit_aer.primitives import EstimatorV2 as AerEstimator
            from qiskit_aer import AerSimulator
            aer_sim = AerSimulator(
                method="matrix_product_state",
                matrix_product_state_max_bond_dimension=mps_bond_dim,
            )
            est = AerEstimator.from_backend(aer_sim)
            simulation = "mps"

        def _energy(x):
            job = est.run([(ansatz, qubit_op, list(x))])
            return float(job.result()[0].data.evs)

        rng = np.random.default_rng(42)
        x0 = rng.uniform(-0.05, 0.05, size=n_params)
        t0 = time.perf_counter()
        opt = _minimize(_energy, x0, method="COBYLA",
                        options={"maxiter": maxiter, "rhobeg": 0.3})
        return {
            "status": "ok",
            "energy": opt.fun + ecore,
            "time": time.perf_counter() - t0,
            "simulation": simulation,
            "n_params": n_params,
            "n_iter": opt.get("nfev", opt.get("nit", 0)),
            "converged": bool(opt.success),
        }
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n  [Qiskit VQE] ERROR: {exc}\n{tb}")
        return {"status": "failed", "error": str(exc), "traceback": tb}


def _run_qiskit_vqe(hf, norb, nelec, mps_bond_dim=64, maxiter=200) -> dict:
    """Run Qiskit VQE on PySCF active-space integrals (standard mode)."""
    try:
        from pyscf import ao2mo
        cas = mcscf.CASCI(hf, norb, nelec); cas.verbose = 0
        h1e, ecore = cas.get_h1eff()
        h2e = ao2mo.restore(1, cas.get_h2eff(), norb)
        return _run_qiskit_vqe_on_integrals(
            h1e, h2e, norb, nelec, ecore,
            mps_bond_dim=mps_bond_dim, maxiter=maxiter)
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
    solvers: set[str] | None = None,
    sqd_samples: int = 200,
    big: bool = False,
    basis_set: str = "sto-3g",
) -> dict:
    """Dehalogenase SN2 — FCI / QSCI / Qoro / Qrunch CI / Qrunch VQE per frame.

    When big=True, Qrunch builds the embedded Hamiltonian (projective embedding
    on the full 27-atom system) and ALL solvers (FCI, QSCI, Qoro, Qrunch CI,
    FAST-VQE) run on that same embedded active-space Hamiltonian.  This enables
    larger active spaces (norb up to ~20) and makes all energies directly
    comparable.
    """
    ALL_SOLVERS = {"fci", "qsci_qiskit", "qsci_qoro", "vqe_qoro", "vqe_qiskit",
                   "qrunch_ci", "qrunch_vqe"}
    if solvers is None:
        solvers = ALL_SOLVERS.copy()
    run_fci         = "fci"         in solvers
    run_qsci_qiskit = "qsci_qiskit" in solvers
    run_qsci_qoro   = "qsci_qoro"   in solvers
    run_vqe_qoro    = "vqe_qoro"    in solvers
    run_vqe_qiskit  = "vqe_qiskit"  in solvers
    run_qrunch_ci   = "qrunch_ci"   in solvers
    run_qrunch_vqe  = "qrunch_vqe"  in solvers
    need_qrunch     = run_qrunch_ci or run_qrunch_vqe

    if big:
        geom_path = DEHALOGENASE_LARGE_XYZ
        embedded_indices = EMBEDDED_ATOM_INDICES_LARGE
    else:
        geom_path = DEHALOGENASE_SMALL_XYZ
        embedded_indices = EMBEDDED_ATOM_INDICES_SMALL

    if not geom_path.exists():
        raise FileNotFoundError(
            f"Geometry not found: {geom_path}\n"
            "Copy the required .xyz file into "
            "benchmarks/geometries/dehalogenase_data/."
        )

    # Parse all frames: full system (for embedding) and embedded subset (for PySCF)
    all_frames_full = _parse_xyz_frames(geom_path)
    all_frames_sub  = _parse_xyz_frames(geom_path, embedded_indices)

    if frame_indices is None:
        frame_indices = DEFAULT_FRAMES
    n_qubits = 2 * norb

    # Qrunch is required in --big mode (for embedding) or when any qrunch solver is selected
    qc, qrunch_skip_reason = _try_import_qrunch() if (need_qrunch or big) else (None, "not selected")
    qrunch_available = qc is not None

    if big and not qrunch_available:
        raise RuntimeError(
            f"--big requires qrunch but it is not available: {qrunch_skip_reason}"
        )

    # Qrunch basis uses no dash (e.g. "sto3g" not "sto-3g")
    qrunch_basis = basis_set.replace("-", "")

    mode_label = "BIG (embedded)" if big else "standard (isolated)"
    solver_list = sorted(solvers & ALL_SOLVERS)
    print(f"\nDehalogenase SN2  CAS({sum(nelec)}e,{norb}o) = {n_qubits}q  "
          f"ansatz={ansatz}  χ={mps_bond_dim}  maxiter={maxiter}")
    print(f"  Mode    : {mode_label}  basis={basis_set}")
    print(f"  Solvers : {', '.join(solver_list)}")
    print(f"  Frames  : {frame_indices}")
    if run_qsci_qiskit or run_qsci_qoro:
        print(f"  QSCI n_samples={sqd_samples} (qsci_qiskit: uniform random; qsci_qoro: circuit-guided)")
    if big:
        print(f"  All solvers use the same Qrunch-embedded Hamiltonian")
    if run_fci:
        print(f"  Errors are relative to FCI (variational error)")
    print(f"  ΔE      = E(frame)   − E(frame 0)   (reaction energy profile)")

    # Build dynamic header
    hdr = f"\n  {'Frame':>5}  "
    if run_fci:                      hdr += f"{'FCI (Ha)':>13} {'t':>6}  "
    if run_qsci_qiskit:              hdr += f"{'QSCI-Qiskit (Ha)':>16} {'t':>6}  "
    if run_fci and run_qsci_qiskit:  hdr += f"{'err_QSCI_Qk':>13}  "
    if run_qsci_qoro:                hdr += f"{'QSCI-Qoro (Ha)':>14} {'t':>6}  "
    if run_fci and run_qsci_qoro:    hdr += f"{'err_QSCI_Qr':>13}  "
    if run_vqe_qoro:                 hdr += f"{'VQE-Qoro (Ha)':>13} {'t':>7}  "
    if run_fci and run_vqe_qoro:     hdr += f"{'err_VQE_Qr':>13}  "
    if run_vqe_qiskit:               hdr += f"{'VQE-Qiskit (Ha)':>15} {'t':>7}  "
    if run_fci and run_vqe_qiskit:   hdr += f"{'err_VQE_Qk':>13}  "
    if run_qrunch_ci:                hdr += f"{'Qrunch CI (Ha)':>14} {'t':>7}  "
    if run_fci and run_qrunch_ci:    hdr += f"{'err_QCI (Ha)':>13}  "
    if run_qrunch_vqe:               hdr += f"{'FAST-VQE (Ha)':>13} {'t':>7}  "
    if run_fci and run_qrunch_vqe:   hdr += f"{'err_VQE (Ha)':>13}"
    print(hdr)

    records       = []
    e_fci0        = None
    e_sqd_qk0     = None
    e_sqd_qr0     = None
    e_m0          = None
    e_qv0         = None
    e_qci0        = None
    e_qvqe0       = None

    for frame_idx in frame_indices:
        if frame_idx >= len(all_frames_full):
            continue

        sub_atoms  = all_frames_sub[frame_idx]
        full_atoms = all_frames_full[frame_idx]

        if big:
            # ── BIG mode: Qrunch builds embedded problem, all solvers use it ──
            print(f"  {frame_idx:5d}  Embedding...", end="", flush=True)
            problem = _build_embedded_problem(
                full_atoms, embedded_indices, norb, nelec[0],
                frame_idx, qc, basis_set=qrunch_basis,
            )
            h1e, h2e, ecore = _extract_integrals(problem)

            e_fci, t_fci = None, None
            if run_fci:
                print(f"  FCI...", end="", flush=True)
                e_fci, t_fci = _run_fci_on_integrals(h1e, h2e, norb, nelec, ecore)

            sqd_qk = {"status": "skipped"}
            if run_qsci_qiskit:
                print(f"  QSCI-Qiskit...", end="", flush=True)
                sqd_qk = _run_qsci_on_integrals(
                    h1e, h2e, norb, nelec, ecore, num_samples=sqd_samples)
            e_sqd_qk = sqd_qk.get("energy")
            t_sqd_qk = sqd_qk.get("time")

            sqd_qr = {"status": "skipped"}
            if run_qsci_qoro:
                print(f"  QSCI-Qoro...", end="", flush=True)
                sqd_qr = _run_qsci_qoro_on_integrals(
                    h1e, h2e, norb, nelec, ecore, ansatz,
                    "gpu" if gpu else "cpu",
                    mps_bond_dim=mps_bond_dim, n_samples=sqd_samples, maxiter=maxiter)
            e_sqd_qr = sqd_qr.get("energy")
            t_sqd_qr = sqd_qr.get("time")

            mr = {"status": "skipped"}
            if run_vqe_qoro:
                print(f"  VQE-Qoro...", end="", flush=True)
                mr = _run_vqe_qoro_on_integrals(
                    h1e, h2e, norb, nelec, ecore, ansatz,
                    "gpu" if gpu else "cpu",
                    mps_bond_dim=mps_bond_dim, maxiter=maxiter)
            e_m = mr.get("energy")
            t_m = mr.get("time")

            qv = {"status": "skipped"}
            if run_vqe_qiskit:
                print(f"  VQE-Qiskit...", end="", flush=True)
                qv = _run_qiskit_vqe_on_integrals(
                    h1e, h2e, norb, nelec, ecore,
                    mps_bond_dim=mps_bond_dim, maxiter=maxiter)
            e_qv = qv.get("energy")
            t_qv = qv.get("time")

            qci = {"status": "skipped"}
            if run_qrunch_ci:
                print(f"  Qrunch CI...", end="", flush=True)
                qci = _run_qrunch_ci(problem, frame_idx, qc)
            e_qci  = qci.get("energy")
            t_qci  = qci.get("time")

            qvqe = {"status": "skipped"}
            if run_qrunch_vqe:
                print(f"  FAST-VQE...", end="", flush=True)
                qvqe = _run_qrunch_vqe(problem, frame_idx, qc)
            e_qvqe = qvqe.get("energy")
            t_qvqe = qvqe.get("time")

        else:
            # ── Standard mode: PySCF on isolated 5-atom subsystem ─────────────
            mol = _mol_from_atoms(sub_atoms, charge=-1, spin=0, basis=basis_set)
            hf, t_hf = _run_hf(mol)

            # Validate active space fits within available MOs
            n_mo = hf.mo_coeff.shape[1]
            n_core = (mol.nelectron - sum(nelec)) // 2
            n_vir = n_mo - n_core - norb
            if n_vir < 0:
                raise ValueError(
                    f"CAS({sum(nelec)}e,{norb}o) does not fit: "
                    f"{n_mo} MOs in {basis_set}, {n_core} frozen core → "
                    f"only {n_mo - n_core} orbitals available for active space. "
                    f"Max norb = {n_mo - n_core}."
                )

            e_fci, t_fci = None, None
            if run_fci:
                print(f"  {frame_idx:5d}  FCI...", end="", flush=True)
                e_fci, t_fci = _run_casci_fci(hf, norb, nelec)
            else:
                print(f"  {frame_idx:5d}", end="", flush=True)

            sqd_qk = {"status": "skipped"}
            if run_qsci_qiskit:
                print(f"  QSCI-Qiskit...", end="", flush=True)
                sqd_qk = _run_qsci(hf, norb, nelec, num_samples=sqd_samples)
            e_sqd_qk = sqd_qk.get("energy")
            t_sqd_qk = sqd_qk.get("time")

            sqd_qr = {"status": "skipped"}
            if run_qsci_qoro:
                print(f"  QSCI-Qoro...", end="", flush=True)
                sqd_qr = _run_qsci_qoro(hf, norb, nelec, ansatz,
                                        "gpu" if gpu else "cpu",
                                        mps_bond_dim=mps_bond_dim,
                                        n_samples=sqd_samples, maxiter=maxiter)
            e_sqd_qr = sqd_qr.get("energy")
            t_sqd_qr = sqd_qr.get("time")

            mr = {"status": "skipped"}
            if run_vqe_qoro:
                print(f"  VQE-Qoro...", end="", flush=True)
                mr  = _run_vqe_qoro(hf, norb, nelec, ansatz,
                                    "gpu" if gpu else "cpu",
                                    mps_bond_dim=mps_bond_dim, maxiter=maxiter)
            e_m = mr.get("energy")
            t_m = mr.get("time")

            qv = {"status": "skipped"}
            if run_vqe_qiskit:
                print(f"  VQE-Qiskit...", end="", flush=True)
                qv = _run_qiskit_vqe(hf, norb, nelec,
                                     mps_bond_dim=mps_bond_dim, maxiter=maxiter)
            e_qv = qv.get("energy")
            t_qv = qv.get("time")

            # Build Qrunch problem once if either qrunch solver is needed
            qrunch_problem = None
            if (run_qrunch_ci or run_qrunch_vqe) and qrunch_available:
                qrunch_problem = _build_standard_problem(
                    sub_atoms, norb, nelec[0], frame_idx, qc)

            qci = {"status": "skipped"}
            if run_qrunch_ci and qrunch_problem is not None:
                print(f"  Qrunch CI...", end="", flush=True)
                qci = _run_qrunch_ci(qrunch_problem, frame_idx, qc)
            e_qci  = qci.get("energy")
            t_qci  = qci.get("time")

            qvqe = {"status": "skipped"}
            if run_qrunch_vqe and qrunch_problem is not None:
                print(f"  FAST-VQE...", end="", flush=True)
                qvqe = _run_qrunch_vqe(qrunch_problem, frame_idx, qc)
            e_qvqe = qvqe.get("energy")
            t_qvqe = qvqe.get("time")

        # Frame-0 references for ΔE
        if e_fci0    is None and e_fci    is not None: e_fci0    = e_fci
        if e_sqd_qk0 is None and e_sqd_qk is not None: e_sqd_qk0 = e_sqd_qk
        if e_sqd_qr0 is None and e_sqd_qr is not None: e_sqd_qr0 = e_sqd_qr
        if e_m0      is None and e_m      is not None: e_m0      = e_m
        if e_qv0     is None and e_qv     is not None: e_qv0     = e_qv
        if e_qci0    is None and e_qci    is not None: e_qci0    = e_qci
        if e_qvqe0   is None and e_qvqe   is not None: e_qvqe0   = e_qvqe

        err_sqd_qk = (e_sqd_qk - e_fci) if (e_sqd_qk is not None and e_fci is not None) else None
        err_sqd_qr = (e_sqd_qr - e_fci) if (e_sqd_qr is not None and e_fci is not None) else None
        err_m      = (e_m      - e_fci) if (e_m      is not None and e_fci is not None) else None
        err_qv     = (e_qv     - e_fci) if (e_qv     is not None and e_fci is not None) else None
        err_qci    = (e_qci    - e_fci) if (e_qci    is not None and e_fci is not None) else None
        err_vqe    = (e_qvqe   - e_fci) if (e_qvqe   is not None and e_fci is not None) else None

        row = f"\r  {frame_idx:5d}  "
        if run_fci:                    row += f"{_fe(e_fci)} {_ft(t_fci)}  "
        if run_qsci_qiskit:            row += f"{_fe(e_sqd_qk)} {_ft(t_sqd_qk)}  "
        if run_fci and run_qsci_qiskit: row += f"{_fe(err_sqd_qk)}  "
        if run_qsci_qoro:              row += f"{_fe(e_sqd_qr)} {_ft(t_sqd_qr)}  "
        if run_fci and run_qsci_qoro:  row += f"{_fe(err_sqd_qr)}  "
        if run_vqe_qoro:               row += f"{_fe(e_m)} {_ft(t_m)}  "
        if run_fci and run_vqe_qoro:   row += f"{_fe(err_m)}  "
        if run_vqe_qiskit:             row += f"{_fe(e_qv):>15} {_ft(t_qv)}  "
        if run_fci and run_vqe_qiskit: row += f"{_fe(err_qv)}  "
        if run_qrunch_ci:              row += f"{_fe(e_qci)} {_ft(t_qci)}  "
        if run_fci and run_qrunch_ci:  row += f"{_fe(err_qci)}  "
        if run_qrunch_vqe:             row += f"{_fe(e_qvqe)} {_ft(t_qvqe)}  "
        if run_fci and run_qrunch_vqe: row += f"{_fe(err_vqe)}"
        print(row)

        records.append({
            "frame":                frame_idx,
            "big_mode":             big,
            "e_fci":                e_fci,      "t_fci":     t_fci,
            "qsci_qiskit":          sqd_qk,     "err_qsci_qiskit_ha": err_sqd_qk,
            "qsci_qoro":            sqd_qr,     "err_qsci_qoro_ha":   err_sqd_qr,
            "vqe_qoro":             mr,         "err_vqe_qoro_ha":    err_m,
            "vqe_qiskit":           qv,         "err_vqe_qiskit_ha":  err_qv,
            "qrunch_ci":            qci,        "err_qrunch_ci_ha":   err_qci,
            "qrunch_vqe":           qvqe,       "err_qrunch_vqe_ha":  err_vqe,
            "delta_e_fci_ha":       (e_fci    - e_fci0)    if (e_fci    and e_fci0)    else None,
            "delta_e_qsci_qk_ha":   (e_sqd_qk - e_sqd_qk0) if (e_sqd_qk and e_sqd_qk0) else None,
            "delta_e_qsci_qr_ha":   (e_sqd_qr - e_sqd_qr0) if (e_sqd_qr and e_sqd_qr0) else None,
            "delta_e_vqe_qoro_ha":  (e_m      - e_m0)      if (e_m      and e_m0)      else None,
            "delta_e_vqe_qiskit_ha":(e_qv     - e_qv0)     if (e_qv     and e_qv0)     else None,
            "delta_e_qci_ha":       (e_qci    - e_qci0)    if (e_qci    and e_qci0)    else None,
            "delta_e_qvqe_ha":      (e_qvqe   - e_qvqe0)   if (e_qvqe   and e_qvqe0)   else None,
        })

    # ── ΔE reaction profile summary ───────────────────────────────────────────
    if len(records) > 1:
        de_hdr = f"  {'Frame':>5}  "
        if run_fci:          de_hdr += f"{'ΔE_FCI':>13}  "
        if run_qsci_qiskit:  de_hdr += f"{'ΔE_QSCI_Qk':>13}  "
        if run_qsci_qoro:    de_hdr += f"{'ΔE_QSCI_Qr':>13}  "
        if run_vqe_qoro:     de_hdr += f"{'ΔE_VQE_Qr':>13}  "
        if run_vqe_qiskit:   de_hdr += f"{'ΔE_VQE_Qk':>14}  "
        if run_qrunch_ci:    de_hdr += f"{'ΔE_Qrunch_CI':>13}  "
        if run_qrunch_vqe:   de_hdr += f"{'ΔE_FAST-VQE':>13}"
        print(f"\n  ΔE reaction profile (Ha, relative to frame {frame_indices[0]}):")
        print(de_hdr)
        for r in records:
            de_row = f"  {r['frame']:5d}  "
            if run_fci:          de_row += f"{_fd(r['e_fci'],                            e_fci0):>13}  "
            if run_qsci_qiskit:  de_row += f"{_fd(r['qsci_qiskit'].get('energy'),        e_sqd_qk0):>13}  "
            if run_qsci_qoro:    de_row += f"{_fd(r['qsci_qoro'].get('energy'),           e_sqd_qr0):>13}  "
            if run_vqe_qoro:     de_row += f"{_fd(r['vqe_qoro'].get('energy'),            e_m0):>13}  "
            if run_vqe_qiskit:   de_row += f"{_fd(r['vqe_qiskit'].get('energy'),          e_qv0):>14}  "
            if run_qrunch_ci:    de_row += f"{_fd(r['qrunch_ci'].get('energy'),           e_qci0):>13}  "
            if run_qrunch_vqe:   de_row += f"{_fd(r['qrunch_vqe'].get('energy'),         e_qvqe0):>13}"
            print(de_row)

    return {
        "name":           "dehalogenase",
        "basis":          basis_set,
        "big_mode":       big,
        "charge":         -1,
        "norb":           norb,
        "nelec":          list(nelec),
        "n_qubits":       n_qubits,
        "ansatz":         ansatz,
        "mps_bond_dim":   mps_bond_dim,
        "frame_indices":  frame_indices,
        "qrunch_available": qrunch_available,
        "sqd_samples":    sqd_samples,
        "records":        records,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dehalogenase SN2 — FCI / QSCI / Qoro / Qrunch CI / Qrunch VQE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  poetry run python benchmarks/bench_dehalogenase.py
  poetry run python benchmarks/bench_dehalogenase.py --frames all
  poetry run python benchmarks/bench_dehalogenase.py --solvers vqe_qoro,qsci_qoro
  poetry run python benchmarks/bench_dehalogenase.py --solvers fci,qrunch_ci
  poetry run python benchmarks/bench_dehalogenase.py --big --norb 14
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
    parser.add_argument("--norb",      type=int, default=None,
                        help=f"Number of active spatial orbitals "
                             f"(default: {NORB} standard, {BIG_NORB} with --big); "
                             "nelec is set to (norb//2, norb//2)")
    parser.add_argument("--sqd-samples", type=int, default=200,
                        help="Bitstring samples for QSCI (default: 200; "
                             "~1000 recovers FCI for this active space)")
    parser.add_argument("--solvers",   type=str, default=None,
                        help="Comma-separated list of solvers to run "
                             "(default: all). Choices: fci, qsci_qiskit, qsci_qoro, "
                             "vqe_qoro, vqe_qiskit, qrunch_ci, qrunch_vqe")
    parser.add_argument("--big",       action="store_true",
                        help="Use 86-atom geometry with Qrunch projective embedding "
                             "for ALL solvers. Defaults: norb=20 (40q), basis=pc-seg-1. "
                             "Use --norb to scale up/down.")
    parser.add_argument("--basis",     type=str, default=None,
                        help=f"Basis set (default: sto-3g standard, {BIG_BASIS} with --big)")
    parser.add_argument("--output",    type=str, default=None)
    args = parser.parse_args()

    if args.frames is None:
        frame_indices = None
    elif args.frames.strip().lower() == "all":
        frame_indices = list(range(11))
    else:
        frame_indices = [int(x) for x in args.frames.split(",")]

    # Parse --solvers
    valid_solvers = {"fci", "qsci_qiskit", "qsci_qoro", "vqe_qoro", "vqe_qiskit",
                     "qrunch_ci", "qrunch_vqe"}
    if args.solvers is not None:
        solver_set = {s.strip().lower() for s in args.solvers.split(",")}
        unknown = solver_set - valid_solvers
        if unknown:
            parser.error(f"Unknown solver(s): {', '.join(unknown)}. "
                         f"Choose from: {', '.join(sorted(valid_solvers))}")
    else:
        solver_set = None  # means "all"

    print("=" * 72)
    print("  DEHALOGENASE SN2 BENCHMARK")

    # Apply --big defaults for norb and basis when not explicitly set
    norb  = args.norb  if args.norb  is not None else (BIG_NORB  if args.big else NORB)
    basis = args.basis if args.basis is not None else (BIG_BASIS if args.big else "sto-3g")
    nelec = (norb // 2, norb // 2)

    print(f"  Mode    : {'BIG (embedded)' if args.big else 'standard (isolated)'}")
    print(f"  GPU     : {'enabled' if args.gpu else 'disabled'}")
    print(f"  χ       : {args.chi}  |  ansatz : {args.ansatz}  |  maxiter : {args.maxiter}")
    print(f"  Frames  : {frame_indices or DEFAULT_FRAMES}")
    print(f"  norb    : {norb}  |  nelec : {sum(nelec)} ({nelec[0]}α,{nelec[1]}β)  |  qubits : {2*norb}")
    print(f"  Basis   : {basis}")
    print(f"  Solvers : {', '.join(sorted(solver_set)) if solver_set else 'all'}")
    print(f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    t0 = time.perf_counter()
    result = bench_dehalogenase(
        gpu=args.gpu,
        frame_indices=frame_indices,
        norb=norb,
        nelec=nelec,
        mps_bond_dim=args.chi,
        ansatz=args.ansatz,
        maxiter=args.maxiter,
        solvers=solver_set,
        sqd_samples=args.sqd_samples,
        big=args.big,
        basis_set=basis,
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
