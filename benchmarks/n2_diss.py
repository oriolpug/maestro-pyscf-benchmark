#!/usr/bin/env python3
# Copyright 2026 Qoro Quantum Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
N₂ Dissociation Benchmark — CAS(10e, 8o), cc-pvdz
===================================================

Compares Maestro UpCCD (MPS, 16 qubits) vs Qiskit PUCCD (statevector, 14q
after ParityMapper(5,5)) vs classical CCSD(T) across three N–N bond lengths.

16 qubits > SV_QUBIT_LIMIT=14, so Maestro uses MPS simulation.
Qiskit ParityMapper(5,5) reduces to 14 qubits for statevector.

Usage
-----
    python benchmarks/n2_diss.py                  # run + plot (default)
    python benchmarks/n2_diss.py --run            # run and save JSON only
    python benchmarks/n2_diss.py --plot           # plot from latest cache
    python benchmarks/n2_diss.py --plot --cache benchmarks/cache/n2_diss_XYZ.json
    python benchmarks/n2_diss.py --gpu            # GPU backend for Maestro
"""

import argparse
import json
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

CACHE_DIR = Path(__file__).parent / "cache"
PLOTS_DIR = Path(__file__).parent / "plots"
CHEM_ACC_MHA = 1.6   # mHa — chemical accuracy threshold for plots

# Maestro statevector crashes at 16q; 14q (H7) is the safe limit.
SV_QUBIT_LIMIT = 14

# ── Benchmark settings ────────────────────────────────────────────────────────

BOND_DISTANCES = [1.098, 1.6, 2.5]
BASIS          = "cc-pvdz"
NORB           = 8
NALPHA, NBETA  = 5, 5
NELEC          = (NALPHA, NBETA)
N_QUBITS       = 2 * NORB   # = 16; Maestro uses MPS, Qiskit parity→14q


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


def _run_ccsdt(hf):
    t0 = time.perf_counter()
    mycc = cc.CCSD(hf); mycc.verbose = 0; mycc.kernel()
    et = mycc.ccsd_t()
    return hf.e_tot + mycc.e_corr + et, time.perf_counter() - t0


def _run_casci_fci(hf, norb, nelec, timeout_s=60):
    import signal
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


def _run_maestro(hf, norb, nelec, ansatz, backend, simulation=None,
                 mps_bond_dim=128, **kwargs) -> dict:
    """Run Maestro VQE (CASCI). Auto-selects statevector/MPS by qubit count."""
    n_qubits = 2 * norb
    if simulation is None:
        simulation = "statevector" if n_qubits <= SV_QUBIT_LIMIT else "mps"
    cas = mcscf.CASCI(hf, norb, nelec); cas.verbose = 0
    kw = dict(ansatz=ansatz, backend=backend, simulation=simulation,
              verbose=False, **kwargs)
    if simulation == "mps":
        kw["mps_bond_dim"] = mps_bond_dim
    cas.fcisolver = MaestroSolver(**kw)
    try:
        t0 = time.perf_counter()
        energy = cas.kernel()[0]
        elapsed = time.perf_counter() - t0
        return {"status": "ok", "energy": energy, "time": elapsed,
                "simulation": simulation,
                "converged": cas.fcisolver.converged,
                "iters": len(cas.fcisolver.energy_history)}
    except Exception as exc:
        return {"status": "failed", "error": str(exc),
                "traceback": traceback.format_exc()}


def _run_qiskit_vqe(hf, norb, nelec, ansatz_type="PUCCD") -> dict:
    """Qiskit VQE via native Qiskit 2.x API (StatevectorEstimator + scipy).

    Uses ParityMapper(num_particles) — reduces 2*norb qubits by 2.
    ecore (nuclear + frozen-core energy) is added back to match total energy.
    """
    try:
        from scipy.optimize import minimize as _minimize
        from pyscf import ao2mo as _ao2mo
        from qiskit.primitives import StatevectorEstimator
        from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
        from qiskit_nature.second_q.mappers import ParityMapper
        from qiskit_nature.second_q.problems import (
            ElectronicBasis, ElectronicStructureProblem)
        from qiskit_nature.second_q.circuit.library import HartreeFock, PUCCD, UCCSD

        if isinstance(nelec, int):
            nalpha = nbeta = nelec // 2
        else:
            nalpha, nbeta = nelec

        cas_obj = mcscf.CASCI(hf, norb, (nalpha, nbeta)); cas_obj.verbose = 0
        h1, ecore = cas_obj.get_h1eff()
        h2 = _ao2mo.restore(1, cas_obj.get_h2eff(), norb)

        hamiltonian = ElectronicEnergy.from_raw_integrals(h1, h2)
        hamiltonian.constants["inactive energy"] = ecore
        problem = ElectronicStructureProblem(hamiltonian)
        problem.basis = ElectronicBasis.MO
        problem.num_spatial_orbitals = norb
        problem.num_particles = (nalpha, nbeta)

        mapper = ParityMapper(num_particles=(nalpha, nbeta))
        # second_q_ops() returns a tuple; [0] is the Hamiltonian
        qubit_op = mapper.map(problem.second_q_ops()[0])

        hf_state = HartreeFock(norb, (nalpha, nbeta), mapper)
        if ansatz_type == "UCCSD":
            ansatz = UCCSD(norb, (nalpha, nbeta), mapper, initial_state=hf_state)
        else:
            ansatz = PUCCD(norb, (nalpha, nbeta), mapper, initial_state=hf_state)

        n_params = ansatz.num_parameters
        estimator = StatevectorEstimator()

        def _energy(params):
            bound = ansatz.assign_parameters(params)
            return float(estimator.run([(bound, qubit_op)]).result()[0].data.evs)

        t0 = time.perf_counter()
        opt = _minimize(_energy, np.zeros(n_params), method="SLSQP",
                        options={"maxiter": 500, "ftol": 1e-10})
        elapsed = time.perf_counter() - t0

        # ParityMapper result is electronic only; add ecore for total energy
        return {"status": "ok", "energy": opt.fun + ecore, "time": elapsed,
                "n_params": n_params, "n_iter": opt.nit,
                "converged": bool(opt.success)}
    except Exception as exc:
        return {"status": "failed", "error": str(exc),
                "traceback": traceback.format_exc()}


def _err(e, ref):
    if e is None or ref is None: return None
    return abs(e - ref) * 1000


def _fmt(e):
    return f"{e:+.6f}" if e is not None else "    N/A   "


def _fmt_err(err):
    return f"{err:.2f}" if err is not None else "N/A"


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(gpu: bool) -> dict:
    """Run N₂ dissociation benchmark and return results dict."""
    records = []

    print(f"\nN₂ Dissociation — CAS({NALPHA+NBETA}e,{NORB}o) = {N_QUBITS}q  basis={BASIS}")
    print(f"Maestro: UpCCD MPS (χ=128)  |  Qiskit: PUCCD statevector (14q after parity)")
    print(f"Bond distances: {BOND_DISTANCES} Å")
    print("-" * 80)

    for r in BOND_DISTANCES:
        print(f"\n--- d = {r} Å ---")

        mol = gto.M(atom=f"N 0 0 0; N 0 0 {r}", basis=BASIS, spin=0, verbose=0)
        hf = _run_hf(mol)
        print(f"  HF      : {hf.e_tot:+.6f} Ha")

        e_ccsdt, t_ccsdt = _run_ccsdt(hf)
        print(f"  CCSD(T) : {_fmt(e_ccsdt)} Ha  ({t_ccsdt:.2f}s)")

        e_fci, t_fci = _run_casci_fci(hf, NORB, NELEC, timeout_s=60)
        if e_fci is not None:
            print(f"  FCI     : {_fmt(e_fci)} Ha  ({t_fci:.2f}s)")
        else:
            print(f"  FCI     : timeout  ({N_QUBITS}q → {2**N_QUBITS:,} dim)")

        ref     = e_fci if e_fci is not None else e_ccsdt
        ref_lbl = "FCI"  if e_fci is not None else "CCSD(T)"

        print(f"  Qiskit PUCCD ...", end="", flush=True)
        qk = _run_qiskit_vqe(hf, NORB, NELEC, "PUCCD")
        e_qk = qk.get("energy") if qk["status"] == "ok" else None
        if qk["status"] == "ok":
            print(f"  {_fmt(e_qk)} Ha  ({qk['time']:.1f}s)  "
                  f"err_{ref_lbl}={_fmt_err(_err(e_qk, ref))} mHa")
        else:
            print(f"  FAILED: {qk.get('error','')[:80]}")

        print(f"  Maestro UpCCD MPS ...", end="", flush=True)
        m = _run_maestro(hf, NORB, NELEC, "upccd", "gpu" if gpu else "cpu",
                         mps_bond_dim=128, maxiter=300)
        e_m = m.get("energy") if m["status"] == "ok" else None
        if m["status"] == "ok":
            print(f"  {_fmt(e_m)} Ha  ({m['time']:.1f}s)  "
                  f"err_{ref_lbl}={_fmt_err(_err(e_m, ref))} mHa  "
                  f"iters={m['iters']}")
        else:
            print(f"  FAILED: {m.get('error','')[:80]}")

        records.append({
            "bond_length":     r,
            "e_hf":            hf.e_tot,
            "e_ccsdt":         e_ccsdt,  "t_ccsdt": t_ccsdt,
            "e_fci":           e_fci,    "t_fci":   t_fci,
            "qiskit":          qk,
            "maestro":         m,
            "err_ccsdt_mha":   _err(e_ccsdt, e_fci),
            "err_qiskit_mha":  _err(e_qk,    e_fci),
            "err_maestro_mha": _err(e_m,     e_fci),
        })

    return {
        "name":           "n2_diss",
        "molecule":       "N2",
        "basis":          BASIS,
        "norb":           NORB,
        "nelec":          list(NELEC),
        "n_qubits":       N_QUBITS,
        "bond_distances": BOND_DISTANCES,
        "records":        records,
    }


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


def _ok(d): return isinstance(d, dict) and d.get("status") == "ok"


def plot_n2_diss(data: dict, out_dir: Path):
    records  = data["records"]
    d_arr    = [r["bond_length"] for r in records]
    norb     = data["norb"]
    nelec    = data["nelec"]
    n_qubits = data["n_qubits"]
    basis    = data["basis"]

    has_fci = any(r.get("e_fci") is not None for r in records)
    ref_lbl = "FCI" if has_fci else "CCSD(T)"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        f"N₂ Dissociation — CAS({sum(nelec)}e,{norb}o) = {n_qubits}q  ({basis})"
        f"   Maestro UpCCD MPS vs Qiskit PUCCD statevector",
        fontweight="bold",
    )

    # ── Panel 1: Potential energy surface ─────────────────────────────────────
    pes_series = [
        ("RHF",           [r["e_hf"]     for r in records]),
        ("CCSD(T)",       [r["e_ccsdt"]  for r in records]),
        ("FCI",           [r.get("e_fci") for r in records]),
        ("Qiskit PUCCD",  [r["qiskit"].get("energy")  if _ok(r["qiskit"])  else None
                           for r in records]),
        ("Maestro UpCCD", [r["maestro"].get("energy") if _ok(r["maestro"]) else None
                           for r in records]),
    ]
    for label, ys in pes_series:
        xs, ys = _filter(d_arr, ys)
        if xs:
            st = _st(label)
            ax1.plot(list(xs), list(ys), marker=st["marker"], linestyle=st["ls"],
                     color=st["color"], label=label, linewidth=2, markersize=6)
    ax1.set_xlabel("Bond length (Å)")
    ax1.set_ylabel("Energy (Ha)")
    ax1.set_title("Potential Energy Surface")
    ax1.legend(fontsize=9)

    # ── Panel 2: Error vs reference (log scale) ────────────────────────────────
    err_series = [
        ("CCSD(T)",       [r.get("err_ccsdt_mha")   for r in records]),
        ("Qiskit PUCCD",  [r.get("err_qiskit_mha")  for r in records]),
        ("Maestro UpCCD", [r.get("err_maestro_mha") for r in records]),
    ]
    for label, ys in err_series:
        ys_safe = [max(y, 1e-6) if y is not None else None for y in ys]
        xs, ys_safe = _filter(d_arr, ys_safe)
        if xs:
            st = _st(label)
            ax2.plot(list(xs), list(ys_safe), marker=st["marker"], linestyle=st["ls"],
                     color=st["color"], label=label, linewidth=2, markersize=6)
    ax2.axhline(CHEM_ACC_MHA, color="gray", ls=":", lw=1.5, label="Chem. accuracy")
    ax2.set_xlabel("Bond length (Å)")
    ax2.set_ylabel(f"Error vs {ref_lbl} (mHa)")
    ax2.set_title(f"Accuracy vs {ref_lbl}")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9)

    # ── Panel 3: Wall-clock time (log scale) ──────────────────────────────────
    time_series = [
        ("CCSD(T)",       [r.get("t_ccsdt") for r in records]),
        ("FCI",           [r.get("t_fci")   for r in records]),
        ("Qiskit PUCCD",  [r["qiskit"].get("time")  if _ok(r["qiskit"])  else None
                           for r in records]),
        ("Maestro UpCCD", [r["maestro"].get("time") if _ok(r["maestro"]) else None
                           for r in records]),
    ]
    for label, ys in time_series:
        xs, ys = _filter(d_arr, ys)
        if xs:
            st = _st(label)
            ax3.plot(list(xs), list(ys), marker=st["marker"], linestyle=st["ls"],
                     color=st["color"], label=label, linewidth=2, markersize=6)
    ax3.set_xlabel("Bond length (Å)")
    ax3.set_ylabel("Wall-clock time (s)")
    ax3.set_title("Computation Time per Geometry")
    ax3.set_yscale("log")
    ax3.legend(fontsize=9)

    fig.tight_layout()
    path = out_dir / "n2_diss.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _latest_cache() -> Path:
    files = sorted(CACHE_DIR.glob("n2_diss_*.json"))
    if not files:
        sys.exit(f"No n2_diss cache found in {CACHE_DIR}. Run with --run first.")
    return files[-1]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="N₂ dissociation benchmark: Maestro vs Qiskit vs classical",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python benchmarks/n2_diss.py                    # run + plot
  python benchmarks/n2_diss.py --run              # save JSON only
  python benchmarks/n2_diss.py --plot             # plot latest cache
  python benchmarks/n2_diss.py --plot --cache benchmarks/cache/n2_diss_XYZ.json
  python benchmarks/n2_diss.py --gpu              # Maestro GPU backend
        """,
    )
    parser.add_argument("--run",        action="store_true", help="Run benchmark and save JSON")
    parser.add_argument("--plot",       action="store_true", help="Plot from cache (skip run)")
    parser.add_argument("--gpu",        action="store_true", help="GPU backend for Maestro")
    parser.add_argument("--cache",      type=str, default=None,
                        help="Path to cache JSON (default: latest n2_diss_*.json)")
    parser.add_argument("--output-dir", type=str, default=str(PLOTS_DIR),
                        help="Directory for output PNGs")
    args = parser.parse_args()

    # Default (no flags): run and plot
    do_run  = args.run  or (not args.run and not args.plot)
    do_plot = args.plot or (not args.run and not args.plot)

    cache_path = None

    if do_run:
        print("=" * 72)
        print("  N₂ DISSOCIATION BENCHMARK")
        print(f"  GPU   : {'enabled' if args.gpu else 'disabled'}")
        print(f"  Date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 72)

        t0 = time.perf_counter()
        data = run_benchmark(gpu=args.gpu)
        elapsed = time.perf_counter() - t0

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_path = CACHE_DIR / f"n2_diss_{ts}.json"
        payload = {
            "meta": {
                "timestamp":    datetime.now().isoformat(),
                "gpu":          args.gpu,
                "total_time_s": round(elapsed, 2),
            },
            "data": data,
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2, cls=_NumpyEncoder)

        print(f"\n{'=' * 72}")
        print(f"  Done in {elapsed:.1f}s")
        print(f"  Results → {cache_path}")
        print("=" * 72)

    if do_plot:
        if cache_path is None:
            cache_path = Path(args.cache) if args.cache else _latest_cache()

        print(f"\nLoading: {cache_path}")
        with open(cache_path) as f:
            payload = json.load(f)

        meta = payload.get("meta", {})
        if meta:
            print(f"  Date  : {meta.get('timestamp', '?')}")
            print(f"  GPU   : {meta.get('gpu', '?')}")
            print(f"  Time  : {meta.get('total_time_s', '?')}s\n")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_n2_diss(payload["data"], out_dir)
        print(f"Plots → {out_dir}")


if __name__ == "__main__":
    main()
