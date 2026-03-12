#!/usr/bin/env python3
# Copyright 2026 Qoro Quantum Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Benchmark Suite — Four Chemistry Benchmarks
============================================

  Case 1  N₂ dissociation profile    CAS(10e,8o) cc-pvdz   3 bond lengths
  Case 2  Cr₂ dimer                  CAS(12e,12o) cc-pvdz  equilibrium (1.68 Å)
  Case 3  Fe₂S₂ cluster              CAS(14e,14o) def2-svp + norb scaling sweep
  Case 4  Fe-Porphine                CAS(22e,22o) cc-pvdz  + norb scaling sweep

Cases 3 & 4 require XYZ geometry files in benchmarks/geometries/.
If absent, an H-chain model replaces the main benchmark and the scaling sweep
uses increasing H-chains — same qualitative story, always runnable.

The norb scaling sweeps demonstrate how FCI time grows O(exp(N)) while
Maestro MPS remains O(poly(N)), making it viable where FCI and Qiskit
statevector become intractable.

Usage
-----
    python benchmarks/benchmarks.py                       # all cases + plot
    python benchmarks/benchmarks.py --case1 --case2       # selective
    python benchmarks/benchmarks.py --run                 # save JSON, no plot
    python benchmarks/benchmarks.py --plot                # plot latest cache
    python benchmarks/benchmarks.py --gpu                 # Maestro GPU backend
    python benchmarks/benchmarks.py --plot --cache benchmarks/cache/benchmarks_XYZ.json
"""

import argparse
import dataclasses
import json
import signal
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

try:
    from scipy.sparse import SparseEfficiencyWarning
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyscf import cc, gto, mcscf, scf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qoro_maestro_pyscf import MaestroSolver

CACHE_DIR     = Path(__file__).parent / "cache"
PLOTS_DIR     = Path(__file__).parent / "plots"
GEO_DIR       = Path(__file__).parent / "geometries"
CHEM_ACC_MHA   = 1.6   # mHa
SV_QUBIT_LIMIT = 25    # Both Maestro and Qiskit switch to MPS above this qubit count


# ── Benchmark configuration (swapped by --small) ─────────────────────────────

@dataclasses.dataclass
class BenchmarkConfig:
    # Case 1: N₂ PES
    n2_distances:       list        # bond lengths to sample
    n2_maxiter:         int         # Maestro VQE maxiter
    n2_qiskit_timeout:  int         # seconds

    # Case 2: Cr₂
    cr2_maxiter:        int
    cr2_fci_timeout:    int
    cr2_qiskit_timeout: int

    # Cases 3 & 4: main single-point benchmark
    main_maxiter:       int
    main_fci_timeout:   int
    main_qiskit_timeout: int

    # Cases 3 & 4: norb scaling sweep
    norb_sweep_3:       list        # Fe₂S₂
    norb_sweep_4:       list        # Fe-Porphine
    sweep_maxiter:      int
    sweep_fci_timeout:  int
    sweep_qiskit_timeout: int

    # Shared MPS settings
    mps_bond_dim:       int


FULL = BenchmarkConfig(
    n2_distances        = [1.098, 1.6, 2.5],
    n2_maxiter          = 300,
    n2_qiskit_timeout   = 300,

    cr2_maxiter         = 300,
    cr2_fci_timeout     = 30,
    cr2_qiskit_timeout  = 300,

    main_maxiter        = 300,
    main_fci_timeout    = 30,
    main_qiskit_timeout = 300,

    norb_sweep_3        = [4, 6, 8, 10, 12, 14],
    norb_sweep_4        = [4, 6, 8, 10, 12, 14, 18, 22],
    sweep_maxiter       = 200,
    sweep_fci_timeout   = 30,
    sweep_qiskit_timeout = 180,

    mps_bond_dim        = 128,
)

SMALL = BenchmarkConfig(
    # Reduced bond lengths and iterations for fast local prototyping
    n2_distances        = [1.098, 2.5],    # 2 geometries instead of 3
    n2_maxiter          = 100,
    n2_qiskit_timeout   = 60,

    cr2_maxiter         = 100,
    cr2_fci_timeout     = 10,
    cr2_qiskit_timeout  = 60,

    main_maxiter        = 100,
    main_fci_timeout    = 10,
    main_qiskit_timeout = 60,

    norb_sweep_3        = [4, 6, 8],       # 3 points instead of 6
    norb_sweep_4        = [4, 6, 8],       # 3 points instead of 8
    sweep_maxiter       = 50,
    sweep_fci_timeout   = 10,
    sweep_qiskit_timeout = 60,

    mps_bond_dim        = 32,
)


# ── JSON encoder ──────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ── Low-level runners ─────────────────────────────────────────────────────────

def _run_hf(mol):
    hf = scf.RHF(mol); hf.verbose = 0; hf.run(); return hf


def _run_ccsdt(hf) -> tuple[float | None, float | None]:
    try:
        t0 = time.perf_counter()
        mycc = cc.CCSD(hf); mycc.verbose = 0; mycc.kernel()
        et = mycc.ccsd_t()
        return hf.e_tot + mycc.e_corr + et, time.perf_counter() - t0
    except Exception:
        return None, None


def _timed(fn, timeout_s: int):
    """Run fn() under SIGALRM; raises TimeoutError on expiry."""
    def _handler(sig, frame): raise TimeoutError
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        return fn()
    finally:
        signal.alarm(0)


def _run_fci(hf, norb, nelec, timeout_s=60) -> tuple[float | None, float | None]:
    def _do():
        t0 = time.perf_counter()
        cas = mcscf.CASCI(hf, norb, nelec); cas.verbose = 0
        return cas.kernel()[0], time.perf_counter() - t0
    try:
        return _timed(_do, timeout_s)
    except (TimeoutError, Exception):
        return None, None


def _run_maestro(hf, norb, nelec, ansatz, backend, simulation=None,
                 mps_bond_dim=64, timeout_s=0, **kwargs) -> dict:
    n_qubits = 2 * norb
    if simulation is None:
        simulation = "statevector" if n_qubits <= SV_QUBIT_LIMIT else "mps"
    cas = mcscf.CASCI(hf, norb, nelec); cas.verbose = 0
    kw = dict(ansatz=ansatz, backend=backend, simulation=simulation,
              verbose=False, **kwargs)
    if simulation == "mps":
        kw["mps_bond_dim"] = mps_bond_dim
    cas.fcisolver = MaestroSolver(**kw)

    def _do():
        t0 = time.perf_counter()
        energy = cas.kernel()[0]
        elapsed = time.perf_counter() - t0
        return {"status": "ok", "energy": energy, "time": elapsed,
                "simulation": simulation,
                "converged": cas.fcisolver.converged,
                "iters": len(cas.fcisolver.energy_history)}

    try:
        return _timed(_do, timeout_s) if timeout_s > 0 else _do()
    except TimeoutError:
        return {"status": "timeout", "error": f">{timeout_s}s"}
    except Exception as exc:
        return {"status": "failed", "error": str(exc),
                "traceback": traceback.format_exc()}


def _run_qiskit_vqe(hf, norb, nelec, ansatz_type="PUCCD", timeout_s=0, mps_bond_dim: int = 64) -> dict:
    """Qiskit VQE via native Qiskit 2.x API (scipy SLSQP).

    ParityMapper(num_particles) reduces 2*norb qubits by 2.
    Uses StatevectorEstimator when n_qubits_mapped <= SV_QUBIT_LIMIT,
    otherwise switches to Aer MPS (qiskit_aer Estimator with
    method='matrix_product_state').
    ecore is added back so the returned energy matches PySCF CASCI total energy.
    """
    def _do():
        from scipy.optimize import minimize as _minimize
        from pyscf import ao2mo as _ao2mo
        from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
        from qiskit_nature.second_q.mappers import ParityMapper
        from qiskit_nature.second_q.problems import (
            ElectronicBasis, ElectronicStructureProblem)
        from qiskit_nature.second_q.circuit.library import HartreeFock, PUCCD, UCCSD

        nalpha, nbeta = nelec if not isinstance(nelec, int) else (nelec // 2, nelec // 2)
        n_qubits_mapped = 2 * norb - 2  # ParityMapper(num_particles) reduces by 2

        cas_obj = mcscf.CASCI(hf, norb, (nalpha, nbeta)); cas_obj.verbose = 0
        h1, ecore = cas_obj.get_h1eff()
        h2 = _ao2mo.restore(1, cas_obj.get_h2eff(), norb)

        hamiltonian = ElectronicEnergy.from_raw_integrals(h1, h2)
        hamiltonian.constants["inactive energy"] = ecore
        problem = ElectronicStructureProblem(hamiltonian)
        problem.basis = ElectronicBasis.MO
        problem.num_spatial_orbitals = norb
        problem.num_particles = (nalpha, nbeta)

        mapper   = ParityMapper(num_particles=(nalpha, nbeta))
        qubit_op = mapper.map(problem.second_q_ops()[0])

        hf_state = HartreeFock(norb, (nalpha, nbeta), mapper)
        ansatz = (UCCSD if ansatz_type == "UCCSD" else PUCCD)(
            norb, (nalpha, nbeta), mapper, initial_state=hf_state)

        n_params = ansatz.num_parameters

        if n_qubits_mapped <= SV_QUBIT_LIMIT:
            from qiskit.primitives import StatevectorEstimator
            sv_est = StatevectorEstimator()

            def _energy(params):
                bound = ansatz.assign_parameters(params)
                return float(sv_est.run([(bound, qubit_op)]).result()[0].data.evs)

            simulation = "statevector"
        else:
            from qiskit_aer.primitives import EstimatorV2 as AerEstimator
            aer_est = AerEstimator()
            aer_est.set_options(method="matrix_product_state", matrix_product_state_max_bond_dimension=mps_bond_dim)
            params_order = list(ansatz.parameters)

            def _energy(params):
                job = aer_est.run([ansatz], [qubit_op], [list(params)])
                return float(job.result().values[0])

            simulation = "mps"

        t0 = time.perf_counter()
        opt = _minimize(_energy, np.zeros(n_params), method="SLSQP",
                        options={"maxiter": 500, "ftol": 1e-10})
        return {"status": "ok", "energy": opt.fun + ecore,
                "time": time.perf_counter() - t0,
                "n_params": n_params, "n_iter": opt.nit,
                "converged": bool(opt.success),
                "simulation": simulation,
                "n_qubits": n_qubits_mapped}

    try:
        return _timed(_do, timeout_s) if timeout_s > 0 else _do()
    except TimeoutError:
        return {"status": "timeout", "error": f">{timeout_s}s"}
    except Exception as exc:
        return {"status": "failed", "error": str(exc),
                "traceback": traceback.format_exc()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _h_chain(n: int, spacing: float = 1.5) -> gto.Mole:
    atoms = "; ".join(f"H 0 0 {i * spacing:.3f}" for i in range(n))
    return gto.M(atom=atoms, basis="sto-3g", verbose=0)


def _err(e, ref):
    return None if e is None or ref is None else abs(e - ref) * 1000


def _fmt(e, p=6):
    return f"{e:+.{p}f}" if e is not None else "    N/A   "


def _ok(d): return isinstance(d, dict) and d.get("status") == "ok"


# ── Case 1: N₂ dissociation ───────────────────────────────────────────────────

def bench_case1_n2(gpu: bool, cfg: BenchmarkConfig) -> dict:
    """N₂ CAS(10e,8o) cc-pvdz — PES across bond lengths."""
    norb, nelec = 8, (5, 5)
    n_qubits    = 2 * norb          # 16q → Maestro MPS; Qiskit parity → 14q
    distances   = cfg.n2_distances
    records     = []

    print(f"\n[Case 1] N₂ Dissociation  CAS({sum(nelec)}e,{norb}o) = {n_qubits}q  cc-pvdz")
    print(f"  bond lengths: {distances}")
    print(f"  {'d(Å)':>6}  {'HF':>12}  {'CCSD(T)':>12}  {'FCI':>12}  "
          f"{'Qiskit':>12}  {'Maestro':>12}  {'err_Qk':>8}  {'err_M':>8}")

    for d in distances:
        mol = gto.M(atom=f"N 0 0 0; N 0 0 {d}", basis="cc-pvdz", spin=0, verbose=0)
        hf  = _run_hf(mol)
        e_ccsdt, t_ccsdt = _run_ccsdt(hf)
        e_fci,   t_fci   = _run_fci(hf, norb, nelec, timeout_s=60)
        ref     = e_fci if e_fci is not None else e_ccsdt
        ref_lbl = "FCI"  if e_fci is not None else "CCSD(T)"

        print(f"  {d:6.3f}  Qiskit...", end="", flush=True)
        qk = _run_qiskit_vqe(hf, norb, nelec, "PUCCD", timeout_s=cfg.n2_qiskit_timeout)
        print(f"  Maestro...", end="", flush=True)
        m  = _run_maestro(hf, norb, nelec, "upccd", "gpu" if gpu else "cpu",
                          maxiter=cfg.n2_maxiter, mps_bond_dim=cfg.mps_bond_dim)

        e_qk = qk.get("energy") if _ok(qk) else None
        e_m  = m.get("energy")  if _ok(m)  else None

        print(f"\r  {d:6.3f}  {_fmt(hf.e_tot)}  {_fmt(e_ccsdt)}  {_fmt(e_fci)}  "
              f"{_fmt(e_qk):>12}  {_fmt(e_m):>12}  "
              f"{_err(e_qk, ref) or 0:>8.2f}  {_err(e_m, ref) or 0:>8.2f}"
              f"  [err vs {ref_lbl}]")

        records.append({
            "bond_length":     d,
            "e_hf":            hf.e_tot,
            "e_ccsdt":         e_ccsdt,  "t_ccsdt": t_ccsdt,
            "e_fci":           e_fci,    "t_fci":   t_fci,
            "qiskit":          qk,
            "maestro":         m,
            "err_ccsdt_mha":   _err(e_ccsdt, e_fci),
            "err_qiskit_mha":  _err(e_qk,    e_fci),
            "err_maestro_mha": _err(e_m,     e_fci),
        })

    return {"name": "n2_diss", "norb": norb, "nelec": list(nelec),
            "n_qubits": n_qubits, "basis": "cc-pvdz",
            "bond_distances": distances, "records": records}


# ── Case 2: Cr₂ dimer ─────────────────────────────────────────────────────────

def bench_case2_cr2(gpu: bool, cfg: BenchmarkConfig) -> dict:
    """Cr₂ CAS(12e,12o) cc-pvdz — single geometry (d = 1.68 Å)."""
    norb, nelec = 12, (6, 6)
    n_qubits    = 2 * norb    # 24q → Maestro MPS; Qiskit parity → 22q (very slow)

    print(f"\n[Case 2] Cr₂ Dimer  CAS({sum(nelec)}e,{norb}o) = {n_qubits}q  cc-pvdz  d=1.68Å")

    mol = gto.M(atom="Cr 0 0 0; Cr 0 0 1.68", basis="cc-pvdz", spin=0, verbose=0)
    hf  = _run_hf(mol)
    print(f"  HF      : {hf.e_tot:+.6f} Ha")

    e_ccsdt, t_ccsdt = _run_ccsdt(hf)
    if e_ccsdt is not None:
        print(f"  CCSD(T) : {e_ccsdt:+.6f} Ha  ({t_ccsdt:.1f}s)")
    else:
        print(f"  CCSD(T) : FAILED (strong correlation)")

    e_fci, t_fci = _run_fci(hf, norb, nelec, timeout_s=cfg.cr2_fci_timeout)
    if e_fci is not None:
        print(f"  FCI     : {e_fci:+.6f} Ha  ({t_fci:.1f}s)")
    else:
        print(f"  FCI     : timeout  ({n_qubits}q → {2**n_qubits:,} dim)")

    print(f"  Qiskit PUCCD ...", end="", flush=True)
    qk = _run_qiskit_vqe(hf, norb, nelec, "PUCCD", timeout_s=cfg.cr2_qiskit_timeout)
    e_qk = qk.get("energy") if _ok(qk) else None
    if _ok(qk):
        print(f"  ok: {_fmt(e_qk)} Ha  ({qk['time']:.1f}s)")
    else:
        print(f"  {qk.get('status', 'FAILED')}: {qk.get('error', '')[:80]}")

    print(f"  Maestro UpCCD MPS ...", end="", flush=True)
    m  = _run_maestro(hf, norb, nelec, "upccd", "gpu" if gpu else "cpu",
                      maxiter=cfg.cr2_maxiter, mps_bond_dim=cfg.mps_bond_dim)
    e_m = m.get("energy") if _ok(m) else None
    if _ok(m):
        print(f"  ok: {_fmt(e_m)} Ha  ({m['time']:.1f}s)  iters={m.get('iters')}")
    else:
        print(f"  {m.get('status', 'FAILED')}: {m.get('error', '')[:80]}")

    ref     = e_fci if e_fci is not None else e_ccsdt
    ref_lbl = "FCI"  if e_fci is not None else "CCSD(T)"

    return {"name": "cr2", "norb": norb, "nelec": list(nelec),
            "n_qubits": n_qubits, "basis": "cc-pvdz",
            "e_hf": hf.e_tot,
            "e_ccsdt": e_ccsdt,  "t_ccsdt": t_ccsdt,
            "e_fci":   e_fci,    "t_fci":   t_fci,
            "ref": ref, "ref_lbl": ref_lbl,
            "qiskit":  qk,       "maestro": m,
            "err_ccsdt_mha":   _err(e_ccsdt, ref),
            "err_qiskit_mha":  _err(e_qk,    ref),
            "err_maestro_mha": _err(e_m,     ref)}


# ── Scaling sweep (shared by Cases 3 & 4) ─────────────────────────────────────

def _run_scaling_sweep(hf_or_builder, norb_values: list[int], gpu: bool,
                       fci_timeout=30, qiskit_timeout=180,
                       maxiter=200, mps_bond_dim=128) -> list[dict]:
    """For each norb, run FCI / Qiskit PUCCD / Maestro UpCCD.

    hf_or_builder: either a pyscf SCF object (shared HF, vary active space)
                   or a callable norb → gto.Mole (build a fresh molecule per norb).
    nelec = (norb//2, norb//2)  [half-filling — maximises entanglement, worst case].
    """
    records = []
    print(f"  {'norb':>4}  {'qubits':>6}  "
          f"{'FCI time':>10}  {'Qiskit time':>12}  {'Maestro time':>13}  notes")

    for norb in norb_values:
        nelec    = (norb // 2, norb // 2)
        n_qubits = 2 * norb

        if isinstance(hf_or_builder, scf.hf.SCF):
            hf = hf_or_builder
        else:
            mol = hf_or_builder(norb)
            hf  = _run_hf(mol)

        e_fci, t_fci = _run_fci(hf, norb, nelec, timeout_s=fci_timeout)
        fci_str = f"{t_fci:.1f}s" if t_fci is not None else f">{fci_timeout}s (TO)"

        qk    = _run_qiskit_vqe(hf, norb, nelec, "PUCCD", timeout_s=qiskit_timeout)
        qk_ok = _ok(qk)
        qk_str = f"{qk['time']:.1f}s" if qk_ok else f">{qiskit_timeout}s ({qk.get('status','?')})"

        m    = _run_maestro(hf, norb, nelec, "upccd", "gpu" if gpu else "cpu",
                            maxiter=maxiter, mps_bond_dim=mps_bond_dim)
        m_ok = _ok(m)
        m_str = f"{m['time']:.1f}s" if m_ok else m.get("status", "?")

        sim = m.get("simulation", "?") if m_ok else "—"
        print(f"  {norb:4d}  {n_qubits:6d}q  {fci_str:>10}  {qk_str:>12}  {m_str:>13}  Maestro={sim}")

        ref = e_fci
        records.append({
            "norb":      norb,
            "n_qubits":  n_qubits,
            "nelec":     list(nelec),
            "e_fci":     e_fci,   "t_fci":  t_fci,   "fci_timeout": e_fci is None,
            "qiskit":    qk,
            "maestro":   m,
            "err_qiskit_mha":  _err(qk.get("energy") if qk_ok else None, ref),
            "err_maestro_mha": _err(m.get("energy")  if m_ok  else None, ref),
        })
    return records


def _bench_with_scaling(case_name: str, geo_path, mol_kwargs: dict,
                        main_norb: int, main_nelec: tuple,
                        norb_sweep: list[int], gpu: bool,
                        cfg: BenchmarkConfig) -> dict:
    """Run a main CASCI benchmark (if geometry available) + norb scaling sweep.

    If geo_path is None or missing: skip main benchmark, use H-chain for sweep.
    """
    mol_available  = geo_path is not None and geo_path.exists()
    hf_or_builder  = _h_chain   # default; overridden below if mol loads successfully

    main_result = None
    if mol_available:
        try:
            mol = gto.M(atom=str(geo_path), verbose=0, **mol_kwargs)
            hf  = _run_hf(mol)
            n_q = 2 * main_norb
            print(f"  HF energy  : {hf.e_tot:+.6f} Ha")

            e_fci, t_fci = _run_fci(hf, main_norb, main_nelec,
                                     timeout_s=cfg.main_fci_timeout)
            if e_fci is not None:
                print(f"  FCI        : {e_fci:+.6f} Ha  ({t_fci:.1f}s)")
            else:
                print(f"  FCI        : timeout  ({n_q}q → {2**n_q:,} dim)")

            print(f"  Qiskit PUCCD ...", end="", flush=True)
            qk = _run_qiskit_vqe(hf, main_norb, main_nelec, "PUCCD",
                                  timeout_s=cfg.main_qiskit_timeout, mps_bond_dim=cfg.mps_bond_dim)
            e_qk = qk.get("energy") if _ok(qk) else None
            if _ok(qk):
                print(f"  ok: {_fmt(e_qk)} Ha  ({qk['time']:.1f}s)")
            else:
                print(f"  {qk.get('status', '?')}: {qk.get('error', '')[:80]}")

            print(f"  Maestro UpCCD ...", end="", flush=True)
            m  = _run_maestro(hf, main_norb, main_nelec, "upccd",
                              "gpu" if gpu else "cpu",
                              maxiter=cfg.main_maxiter, mps_bond_dim=cfg.mps_bond_dim)
            e_m = m.get("energy") if _ok(m) else None
            if _ok(m):
                print(f"  ok: {_fmt(e_m)} Ha  ({m['time']:.1f}s)  iters={m.get('iters')}")
            else:
                print(f"  {m.get('status', '?')}: {m.get('error', '')[:80]}")

            ref = e_fci
            main_result = {
                "e_hf":   hf.e_tot,
                "e_fci":  e_fci,  "t_fci": t_fci,
                "qiskit": qk,     "maestro": m,
                "err_qiskit_mha":  _err(e_qk, ref),
                "err_maestro_mha": _err(e_m,  ref),
            }

            # Scaling sweep on the real molecule (vary active space size)
            print(f"\n  Scaling sweep on {geo_path.name}: norb ∈ {norb_sweep}")
            hf_or_builder = hf
        except Exception as exc:
            print(f"  ERROR loading geometry: {exc}")
            mol_available = False

    if not mol_available:
        label = geo_path.name if geo_path else "geometry"
        print(f"  '{label}' not found — using H-chain model for scaling sweep")
        hf_or_builder = _h_chain   # callable: norb → Mole

    if mol_available and main_result is not None:
        pass   # hf_or_builder already set above
    elif not mol_available:
        print(f"  Scaling sweep (H-chain, STO-3G): norb ∈ {norb_sweep}")

    sweep = _run_scaling_sweep(
        hf_or_builder, norb_sweep, gpu,
        fci_timeout=cfg.sweep_fci_timeout,
        qiskit_timeout=cfg.sweep_qiskit_timeout,
        maxiter=cfg.sweep_maxiter,
        mps_bond_dim=cfg.mps_bond_dim,
    )

    return {
        "name":              case_name,
        "main_norb":         main_norb,
        "main_nelec":        list(main_nelec),
        "basis":             mol_kwargs.get("basis", "?"),
        "mol_available":     mol_available,
        "norb_sweep":        norb_sweep,
        "main":              main_result,
        "sweep":             sweep,
        # Store timeout caps so the plot can annotate ▼ at the right level
        "sweep_fci_timeout": cfg.sweep_fci_timeout,
        "sweep_qk_timeout":  cfg.sweep_qiskit_timeout,
    }


def bench_case3_fe2s2(gpu: bool, cfg: BenchmarkConfig) -> dict:
    """Fe₂S₂ cluster: CAS(14e,14o) def2-svp + norb scaling sweep."""
    geo = GEO_DIR / "fe2s2_cluster.xyz"
    print(f"\n[Case 3] Fe₂S₂ Cluster  CAS(14e,14o) = 28q  def2-svp")
    print(f"  Geometry: {'found' if geo.exists() else 'NOT FOUND — H-chain fallback'}")
    return _bench_with_scaling(
        "fe2s2", geo,
        mol_kwargs={"basis": "def2-svp", "charge": -2, "spin": 0},
        main_norb=14, main_nelec=(7, 7),
        norb_sweep=cfg.norb_sweep_3,
        gpu=gpu, cfg=cfg,
    )


def bench_case4_feporphine(gpu: bool, cfg: BenchmarkConfig) -> dict:
    """Fe-Porphine: CAS(22e,22o) cc-pvdz + norb scaling sweep."""
    geo = GEO_DIR / "fe_porphine.xyz"
    print(f"\n[Case 4] Fe-Porphine  CAS(22e,22o) = 44q  cc-pvdz")
    print(f"  Geometry: {'found' if geo.exists() else 'NOT FOUND — H-chain fallback'}")
    return _bench_with_scaling(
        "feporphine", geo,
        mol_kwargs={"basis": "cc-pvdz", "charge": 0, "spin": 2},
        main_norb=22, main_nelec=(11, 11),
        norb_sweep=cfg.norb_sweep_4,
        gpu=gpu, cfg=cfg,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

STYLE = {
    "RHF":           {"color": "#e74c3c", "marker": "^", "ls": ":"},
    "CCSD":          {"color": "#e67e22", "marker": "s", "ls": "--"},
    "CCSD(T)":       {"color": "#f39c12", "marker": "D", "ls": "--"},
    "FCI":           {"color": "#27ae60", "marker": "*", "ls": "-"},
    "Qiskit PUCCD":  {"color": "#8e44ad", "marker": "P", "ls": "-."},
    "Maestro UpCCD": {"color": "#2980b9", "marker": "o", "ls": "-"},
}


def _st(key):
    return STYLE.get(key, {"color": "#555555", "marker": "o", "ls": "-"})


def _filter(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs: return [], []
    return zip(*pairs)


# ── Plot annotation helpers ───────────────────────────────────────────────────

def _annotate_times(ax, xs, ys, color):
    """Label each data point with its elapsed time in seconds."""
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}s", (x, y),
                    textcoords="offset points", xytext=(4, 5),
                    ha="left", fontsize=7, color=color, alpha=0.85,
                    clip_on=False)


def _add_pct_axis(ax, e_ref_ha):
    """Add a right y-axis showing relative error in % using a fixed E_ref.

    rel_err% = err_mHa / (1000 * |E_ref_Ha|) * 100
             = err_mHa / (10 * |E_ref_Ha|)

    Calls ax.autoscale() to ensure limits are finalised before creating twin.
    Labelled 'Rel. error (%)' — approximate when E_ref varies across x-points.
    """
    if not e_ref_ha:
        return
    ax.autoscale()
    y1, y2 = ax.get_ylim()
    scale = 1.0 / (10.0 * abs(e_ref_ha))
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(y1 * scale, y2 * scale)
    ax2.set_ylabel("Rel. error (%)", fontsize=9)
    ax2.tick_params(which="both", labelsize=8)


# ── Case 1 plot ───────────────────────────────────────────────────────────────

def plot_case1(data: dict, out_dir: Path):
    records = data["records"]
    d_arr   = [r["bond_length"] for r in records]
    norb, nelec, basis = data["norb"], data["nelec"], data["basis"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        f"N₂ Dissociation  CAS({sum(nelec)}e,{norb}o) = {data['n_qubits']}q  {basis}",
        fontweight="bold")

    # PES
    for label, ys in [
        ("RHF",           [r["e_hf"]    for r in records]),
        ("CCSD(T)",       [r["e_ccsdt"] for r in records]),
        ("FCI",           [r.get("e_fci") for r in records]),
        ("Qiskit PUCCD",  [r["qiskit"].get("energy")  if _ok(r["qiskit"])  else None for r in records]),
        ("Maestro UpCCD", [r["maestro"].get("energy") if _ok(r["maestro"]) else None for r in records]),
    ]:
        xs, ys = _filter(d_arr, ys)
        if xs:
            st = _st(label)
            ax1.plot(list(xs), list(ys), marker=st["marker"], ls=st["ls"],
                     color=st["color"], label=label, lw=2, ms=6)
    ax1.set(xlabel="Bond length (Å)", ylabel="Energy (Ha)", title="Potential Energy Surface")
    ax1.legend(fontsize=9)

    # Error vs FCI / CCSD(T)
    has_fci = any(r.get("e_fci") is not None for r in records)
    ref_lbl = "FCI" if has_fci else "CCSD(T)"
    for label, key in [("CCSD(T)", "err_ccsdt_mha"),
                        ("Qiskit PUCCD", "err_qiskit_mha"),
                        ("Maestro UpCCD", "err_maestro_mha")]:
        ys = [max(r.get(key) or 0, 1e-6) if r.get(key) is not None else None for r in records]
        xs, ys = _filter(d_arr, ys)
        if xs:
            st = _st(label)
            ax2.plot(list(xs), list(ys), marker=st["marker"], ls=st["ls"],
                     color=st["color"], label=label, lw=2, ms=6)
    ax2.axhline(CHEM_ACC_MHA, color="gray", ls=":", lw=1.5, label="Chem. accuracy")
    ax2.set(xlabel="Bond length (Å)", ylabel=f"Error vs {ref_lbl} (mHa)",
            title=f"Accuracy vs {ref_lbl}", yscale="log")
    ax2.legend(fontsize=9)
    # Right axis: relative error in % (uses mean reference energy as fixed scale)
    e_refs = [r.get("e_fci") if r.get("e_fci") is not None else r.get("e_ccsdt")
              for r in records]
    e_refs = [e for e in e_refs if e is not None]
    _add_pct_axis(ax2, sum(e_refs) / len(e_refs) if e_refs else None)

    # Timing — annotate each point with elapsed seconds
    for label, ys in [
        ("CCSD(T)",       [r.get("t_ccsdt") for r in records]),
        ("FCI",           [r.get("t_fci")   for r in records]),
        ("Qiskit PUCCD",  [r["qiskit"].get("time")  if _ok(r["qiskit"])  else None for r in records]),
        ("Maestro UpCCD", [r["maestro"].get("time") if _ok(r["maestro"]) else None for r in records]),
    ]:
        xs, ys = _filter(d_arr, ys)
        if xs:
            xs_l, ys_l = list(xs), list(ys)
            st = _st(label)
            ax3.plot(xs_l, ys_l, marker=st["marker"], ls=st["ls"],
                     color=st["color"], label=label, lw=2, ms=6)
            _annotate_times(ax3, xs_l, ys_l, st["color"])
    ax3.set(xlabel="Bond length (Å)", ylabel="Wall-clock time (s)",
            title="Computation Time", yscale="log")
    ax3.legend(fontsize=9)

    fig.tight_layout()
    path = out_dir / "case1_n2_diss.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ── Case 2 plot ───────────────────────────────────────────────────────────────

def plot_case2(data: dict, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Cr₂ Dimer  CAS({sum(data['nelec'])}e,{data['norb']}o) = {data['n_qubits']}q  "
        f"{data['basis']}  (d = 1.68 Å)",
        fontweight="bold")

    # Energy bar chart
    energy_pairs = []
    for label, key in [("HF", "e_hf"), ("CCSD(T)", "e_ccsdt"), ("FCI", "e_fci")]:
        e = data.get(key)
        if e is not None:
            energy_pairs.append((label, e))
    for label, key in [("Qiskit PUCCD", "qiskit"), ("Maestro UpCCD", "maestro")]:
        d = data.get(key, {})
        if _ok(d):
            energy_pairs.append((label, d["energy"]))

    if energy_pairs:
        lbls  = [p[0] for p in energy_pairs]
        engs  = [p[1] for p in energy_pairs]
        cols  = [_st(l)["color"] for l in lbls]
        bars  = ax1.bar(lbls, engs, color=cols, edgecolor="black", linewidth=0.5)
        ax1.set(ylabel="Energy (Ha)", title="Total Energy Comparison")
        ax1.tick_params(axis="x", rotation=30)
        for bar, e in zip(bars, engs):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{e:.4f}", ha="center", va="bottom", fontsize=8)

    # Timing bar chart
    time_pairs = []
    for label, t_key in [("CCSD(T)", "t_ccsdt"), ("FCI", "t_fci")]:
        t = data.get(t_key)
        if t is not None:
            time_pairs.append((label, t))
    for label, key in [("Qiskit PUCCD", "qiskit"), ("Maestro UpCCD", "maestro")]:
        d = data.get(key, {})
        if _ok(d):
            time_pairs.append((label, d["time"]))

    if time_pairs:
        t_lbls = [p[0] for p in time_pairs]
        times  = [p[1] for p in time_pairs]
        t_cols = [_st(l)["color"] for l in t_lbls]
        bars = ax2.bar(t_lbls, times, color=t_cols, edgecolor="black", linewidth=0.5)
        ax2.set(ylabel="Wall-clock time (s)", title="Computation Time", yscale="log")
        ax2.tick_params(axis="x", rotation=30)
        for bar, t in zip(bars, times):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{t:.1f}s", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = out_dir / "case2_cr2.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ── Cases 3 & 4 scaling plot ──────────────────────────────────────────────────

def plot_scaling(data: dict, out_dir: Path, filename: str):
    """2-panel scaling plot: time vs norb (log) + accuracy vs norb (log).

    Left panel is the headline: shows FCI growing O(exp(N)) with timeout markers
    (▼), Qiskit also eventually timing out, while Maestro MPS remains tractable.
    """
    sweep       = data["sweep"]
    mol_lbl     = {"fe2s2": "Fe₂S₂ Cluster", "feporphine": "Fe-Porphine"}.get(data["name"], data["name"])
    sweep_source = "real molecule" if data["mol_available"] else "H-chain model"
    main        = data.get("main")
    main_norb   = data["main_norb"]
    main_nelec  = data["main_nelec"]

    n_panels = 3 if main is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5))
    if n_panels == 1: axes = [axes]

    fig.suptitle(
        f"{mol_lbl}  CAS({sum(main_nelec)}e,{main_norb}o)  {data['basis']}\n"
        f"Scaling sweep: norb ∈ {data['norb_sweep']}  ({sweep_source})",
        fontweight="bold")

    norb_arr = [r["norb"]     for r in sweep]
    q_arr    = [r["n_qubits"] for r in sweep]

    # ── Panel 1: Time vs norb (log scale) ─────────────────────────────────────
    ax_t = axes[0]

    # FCI — plot available times, mark timeouts with ▼ at cap
    fci_t_cap = data.get("sweep_fci_timeout", 30)
    qk_t_cap  = data.get("sweep_qk_timeout",  180)
    for label, times, cap in [
        ("FCI",           [r["t_fci"]                                  for r in sweep], fci_t_cap),
        ("Qiskit PUCCD",  [r["qiskit"].get("time") if _ok(r["qiskit"]) else None for r in sweep], qk_t_cap),
        ("Maestro UpCCD", [r["maestro"].get("time") if _ok(r["maestro"]) else None for r in sweep], None),
    ]:
        xs, ys = _filter(norb_arr, times)
        if xs:
            xs_l, ys_l = list(xs), list(ys)
            st = _st(label)
            ax_t.semilogy(xs_l, ys_l, marker=st["marker"], ls=st["ls"],
                          color=st["color"], label=label, lw=2, ms=7)
            _annotate_times(ax_t, xs_l, ys_l, st["color"])
        if cap is not None:
            for norb, t in zip(norb_arr, times):
                if t is None:
                    st = _st(label)
                    ax_t.semilogy(norb, cap, marker="v", color=st["color"],
                                  ms=10, ls="none", zorder=5, alpha=0.8)

    # Reference exponential line: 2^(norb/2) scaled to pass through first FCI point
    fci_anchor = next(((r["norb"], r["t_fci"]) for r in sweep if r["t_fci"] is not None), None)
    if fci_anchor:
        n0, t0 = fci_anchor
        ref_xs = [n for n in norb_arr if n >= n0]
        ref_ys = [t0 * 2 ** ((n - n0) / 2) for n in ref_xs]
        ax_t.semilogy(ref_xs, ref_ys, color="gray", ls=":", lw=1.2,
                      label="∝ 2^(norb/2)  [FCI theory]", alpha=0.7)

    ax_t.set_xlabel("Active orbitals (norb)")
    ax_t.set_ylabel("Wall-clock time (s)")
    ax_t.set_title("Scaling: Time vs Active Space Size\n(▼ = timeout)")
    ax_t.set_xticks(norb_arr)
    ax_t.legend(fontsize=9)

    # Secondary top axis: qubit count
    ax_top = ax_t.twiny()
    ax_top.set_xlim(ax_t.get_xlim())
    ax_top.set_xticks(norb_arr)
    ax_top.set_xticklabels([str(q) for q in q_arr], fontsize=8)
    ax_top.set_xlabel("Qubits (2 × norb)", fontsize=9)

    # ── Panel 2: Accuracy vs norb (where FCI available as reference) ──────────
    ax_e = axes[1]

    for label, errs in [
        ("Qiskit PUCCD",  [r.get("err_qiskit_mha")  for r in sweep]),
        ("Maestro UpCCD", [r.get("err_maestro_mha") for r in sweep]),
    ]:
        st = _st(label)
        ys_safe = [max(y, 1e-6) if y is not None else None for y in errs]
        xs, ys_safe = _filter(norb_arr, ys_safe)
        if xs:
            ax_e.semilogy(list(xs), list(ys_safe), marker=st["marker"], ls=st["ls"],
                          color=st["color"], label=label, lw=2, ms=7)
        # Annotate with relative error in % (per-point, using that norb's FCI as ref)
        # E_ref varies with system size so we compute per-point rather than a twin axis
        for r, err_mha in zip(sweep, errs):
            e_ref = r.get("e_fci")
            if err_mha is not None and err_mha > 0 and e_ref is not None:
                pct = err_mha / (10.0 * abs(e_ref))
                ax_e.annotate(f"{pct:.2g}%", (r["norb"], max(err_mha, 1e-6)),
                              textcoords="offset points", xytext=(4, 4),
                              ha="left", fontsize=7, color=st["color"], alpha=0.85,
                              clip_on=False)

    ax_e.axhline(CHEM_ACC_MHA, color="gray", ls=":", lw=1.5, label="Chem. accuracy")
    ax_e.set_xlabel("Active orbitals (norb)")
    ax_e.set_ylabel("Error vs FCI (mHa)")
    ax_e.set_title("Accuracy vs Active Space Size\n(only where FCI converged)")
    ax_e.set_xticks(norb_arr)
    ax_e.legend(fontsize=9)

    # ── Optional panel 3: main single-point energy comparison ─────────────────
    if n_panels == 3 and main is not None:
        ax_m = axes[2]
        pairs = []
        if main.get("e_hf")  is not None: pairs.append(("HF",           main["e_hf"]))
        if main.get("e_fci") is not None: pairs.append(("FCI",          main["e_fci"]))
        if _ok(main.get("qiskit",  {})):  pairs.append(("Qiskit PUCCD", main["qiskit"]["energy"]))
        if _ok(main.get("maestro", {})):  pairs.append(("Maestro UpCCD",main["maestro"]["energy"]))

        if pairs:
            lbls = [p[0] for p in pairs]
            engs = [p[1] for p in pairs]
            cols = [_st(l)["color"] for l in lbls]
            ax_m.bar(lbls, engs, color=cols, edgecolor="black", linewidth=0.5)
            ax_m.set(ylabel="Energy (Ha)",
                     title=f"CAS({sum(main_nelec)}e,{main_norb}o) Energy Comparison")
            ax_m.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _latest_cache() -> Path:
    files = sorted(CACHE_DIR.glob("benchmarks_*.json"))
    if not files:
        sys.exit(f"No benchmark cache in {CACHE_DIR}. Run without --plot first.")
    return files[-1]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chemistry benchmarks: Maestro vs Qiskit vs PySCF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
cases
  --case1   N₂ dissociation  CAS(10e,8o) cc-pvdz       PES, 3 bond lengths
  --case2   Cr₂ dimer        CAS(12e,12o) cc-pvdz      single geometry (1.68 Å)
  --case3   Fe₂S₂ cluster    CAS(14e,14o) def2-svp     main + norb scaling sweep
  --case4   Fe-Porphine      CAS(22e,22o) cc-pvdz      main + norb scaling sweep

modes
  --small   Fast prototyping mode (2 geometries, 3 norb points, χ=32, maxiter=50-100)
            Runs in minutes on a laptop; use full mode on a cluster.

geometry files (cases 3, 4; H-chain fallback used if absent)
  benchmarks/geometries/fe2s2_cluster.xyz
  benchmarks/geometries/fe_porphine.xyz

examples
  python benchmarks/benchmarks.py --small                 # quick smoke-test, all cases
  python benchmarks/benchmarks.py --small --case1         # single case, fast
  python benchmarks/benchmarks.py                         # full run + plot (cluster)
  python benchmarks/benchmarks.py --case1 --case2         # selective + plot
  python benchmarks/benchmarks.py --run --case3 --case4   # run only, no plot
  python benchmarks/benchmarks.py --plot                  # plot latest cache
  python benchmarks/benchmarks.py --gpu --case3           # GPU backend
        """,
    )
    parser.add_argument("--run",        action="store_true", help="Run and save JSON")
    parser.add_argument("--plot",       action="store_true", help="Plot from cache (skip run)")
    parser.add_argument("--small",      action="store_true",
                        help="Fast prototyping: fewer geometries, smaller sweeps, χ=32")
    parser.add_argument("--no-timeout", action="store_true",
                        help="Remove Qiskit VQE time limits (let it run to convergence)")
    parser.add_argument("--gpu",        action="store_true", help="Maestro GPU backend")
    parser.add_argument("--case1",      action="store_true", help="N₂ dissociation")
    parser.add_argument("--case2",      action="store_true", help="Cr₂ dimer")
    parser.add_argument("--case3",      action="store_true", help="Fe₂S₂ + scaling")
    parser.add_argument("--case4",      action="store_true", help="Fe-Porphine + scaling")
    parser.add_argument("--cache",      type=str, default=None,
                        help="Cache JSON path (default: latest benchmarks_*.json)")
    parser.add_argument("--output-dir", type=str, default=str(PLOTS_DIR),
                        help="Directory for output PNGs")
    args = parser.parse_args()

    # Default (no mode flags): run + plot
    do_run  = args.run  or (not args.run and not args.plot)
    do_plot = args.plot or (not args.run and not args.plot)

    cfg    = SMALL if args.small else FULL
    mode   = "small" if args.small else "full"
    if args.no_timeout:
        cfg = dataclasses.replace(
            cfg,
            n2_qiskit_timeout    = 0,
            cr2_qiskit_timeout   = 0,
            main_qiskit_timeout  = 0,
            sweep_qiskit_timeout = 0,
        )

    selected = [k for k, f in [("case1", args.case1), ("case2", args.case2),
                                ("case3", args.case3), ("case4", args.case4)] if f]
    to_run = selected or ["case1", "case2", "case3", "case4"]

    cache_path = None

    if do_run:
        print("=" * 72)
        print("  MAESTRO vs QISKIT vs PYSCF — BENCHMARK SUITE")
        print(f"  Mode    : {mode}  (--small for fast prototyping, full for cluster)")
        print(f"  GPU     : {'enabled' if args.gpu else 'disabled'}")
        print(f"  Cases   : {', '.join(to_run)}")
        if args.small:
            print(f"  Config  : distances={cfg.n2_distances}  "
                  f"norb_sweep_3={cfg.norb_sweep_3}  χ={cfg.mps_bond_dim}  "
                  f"maxiter≤{cfg.n2_maxiter}")
        print(f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 72)

        dispatch = {
            "case1": lambda: bench_case1_n2(args.gpu, cfg),
            "case2": lambda: bench_case2_cr2(args.gpu, cfg),
            "case3": lambda: bench_case3_fe2s2(args.gpu, cfg),
            "case4": lambda: bench_case4_feporphine(args.gpu, cfg),
        }

        results = {
            "meta": {
                "timestamp":  datetime.now().isoformat(),
                "gpu":        args.gpu,
                "mode":       mode,
                "cases_run":  to_run,
            },
            "benchmarks": {},
        }

        t0 = time.perf_counter()
        for name in to_run:
            try:
                results["benchmarks"][name] = dispatch[name]()
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"\n  ERROR in {name}: {exc}\n{tb}")
                results["benchmarks"][name] = {
                    "name": name, "error": str(exc), "traceback": tb}

        results["meta"]["total_time_s"] = round(time.perf_counter() - t0, 2)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_path = CACHE_DIR / f"benchmarks_{ts}.json"
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2, cls=_NumpyEncoder)

        print(f"\n{'=' * 72}")
        print(f"  Done in {results['meta']['total_time_s']:.1f}s")
        print(f"  Results → {cache_path}")
        print(f"  Plots   → python benchmarks/benchmarks.py --plot --cache {cache_path}")
        print("=" * 72)

    if do_plot:
        if cache_path is None:
            cache_path = Path(args.cache) if args.cache else _latest_cache()

        print(f"\nLoading : {cache_path}")
        with open(cache_path) as f:
            all_data = json.load(f)

        meta = all_data.get("meta", {})
        print(f"  Date  : {meta.get('timestamp', '?')}")
        print(f"  Mode  : {meta.get('mode', 'full')}")
        print(f"  GPU   : {meta.get('gpu', '?')}")
        print(f"  Cases : {', '.join(meta.get('cases_run', []))}\n")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_map = {
            "case1": lambda d: plot_case1(d, out_dir),
            "case2": lambda d: plot_case2(d, out_dir),
            "case3": lambda d: plot_scaling(d, out_dir, "case3_fe2s2_scaling.png"),
            "case4": lambda d: plot_scaling(d, out_dir, "case4_feporphine_scaling.png"),
        }
        for key, fn in plot_map.items():
            d = all_data["benchmarks"].get(key)
            if d is None:
                continue
            if "error" in d:
                print(f"  Skipping {key}: {d['error']}")
            else:
                fn(d)

        print(f"\nAll plots → {out_dir}")


if __name__ == "__main__":
    main()
