import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip().split("\n")})

def code(source):
    lines = source.strip().split("\n")
    # add newlines to all but last
    lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "code", "metadata": {}, "source": lines, "outputs": [], "execution_count": None})

md("""# Full Workflow — BeH₂
## Molecule → HF → CASCI + VQE → Properties

A complete computational chemistry pipeline using **qoro-maestro-pyscf**.

This notebook walks through each step of a real quantum chemistry study on BeH₂,
a molecule with interesting multi-reference character due to the near-degeneracy
of the Be 2s and 2p orbitals.""")

code("""import time
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, mcscf, mrpt

from qoro_maestro_pyscf import MaestroSolver
from qoro_maestro_pyscf.properties import (
    compute_dipole_moment,
    compute_natural_orbitals,
)

# Dark theme for plots
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
})""")

md("""## Step 1 — Molecule Definition

BeH₂ in a linear geometry with the STO-3G basis set.""")

code("""mol = gto.M(
    atom=\"\"\"
        Be  0.000  0.000  0.000
        H   0.000  0.000  1.334
        H   0.000  0.000 -1.334
    \"\"\",
    basis="sto-3g",
    symmetry=False,
    verbose=0,
)
print(f"Molecule: BeH₂ (STO-3G)")
print(f"Atoms: {mol.natm}, AOs: {mol.nao_nr()}")""")

md("""## Step 2 — Hartree-Fock

Starting point: a single-determinant mean-field solution.""")

code("""hf = scf.RHF(mol).run()
print(f"E(HF) = {hf.e_tot:+.10f} Ha")""")

md("""## Step 3 — CASCI + VQE on Maestro

We use a (2e, 3o) active space → 6 qubits. The UCCSD ansatz captures
the static correlation that Hartree-Fock misses.

**CASCI** (not CASSCF) is used here because VQE runs once on fixed HF
orbitals, avoiding the convergence issues of CASSCF macro-iterations.""")

code("""norb = 3   # Be 2s, 2pz + H 1s orbitals
nelec = 2
backend = "cpu"

cas = mcscf.CASCI(hf, norb, nelec)
cas.fcisolver = MaestroSolver(
    ansatz="uccsd",
    backend=backend,
    maxiter=300,
    verbose=True,
)

t0 = time.perf_counter()
casci_e = cas.kernel()[0]
casci_time = time.perf_counter() - t0

print(f"\\nE(CASCI+VQE) = {casci_e:+.10f} Ha  ({casci_time:.1f}s)")""")

md("""## Step 4 — NEVPT2 (Perturbation Theory)

NEVPT2 adds dynamic correlation on top of the CASCI solution.

> **Note:** NEVPT2 requires a 3-body reduced density matrix (3-RDM),
> which needs a CI vector in FCI format. VQE solvers produce parameterised
> circuits instead, so NEVPT2 is skipped here. This is a known limitation
> shared by all VQE-based solvers.""")

code("""t0 = time.perf_counter()
try:
    nevpt2_corr = mrpt.NEVPT(cas).kernel()
    nevpt2_e = casci_e + nevpt2_corr
    nevpt2_time = time.perf_counter() - t0
    print(f"ΔE(NEVPT2) = {nevpt2_corr:+.10f} Ha")
    print(f"E(CASCI+NEVPT2) = {nevpt2_e:+.10f} Ha  ({nevpt2_time:.1f}s)")
except (AssertionError, RuntimeError, TypeError):
    nevpt2_e = None
    print("⚠ NEVPT2 requires a 3-RDM from the CI vector.")
    print("VQE solvers produce circuits, not CI vectors — skipping.")""")

md("""## Step 5 — Properties

Extract molecular properties from the VQE wavefunction:
- **Dipole moment** — should be ~0 for symmetric BeH₂
- **Natural orbital occupations** — show correlation effects""")

code("""ci = cas.ci
rdm1 = cas.fcisolver.make_rdm1(ci, norb, (1, 1))

# Dipole moment
dipole, mag = compute_dipole_moment(mol, cas.mo_coeff, rdm1)
print(f"Dipole: [{dipole[0]:+.4f}, {dipole[1]:+.4f}, {dipole[2]:+.4f}] D")
print(f"|μ| = {mag:.4f} D  (should be ~0 by symmetry)")

# Natural orbital occupations
occ, _ = compute_natural_orbitals(rdm1)
print(f"Natural occupations: [{', '.join(f'{n:.4f}' for n in occ)}]")""")

md("""## Summary & Plots

Compare all methods against the exact FCI solution.""")

code("""# FCI reference
cas_fci = mcscf.CASCI(hf, norb, nelec)
cas_fci.verbose = 0
fci_e = cas_fci.kernel()[0]

# Build comparison data (skip NEVPT2 if unavailable)
methods = ['HF', 'CASCI+VQE\\n(Maestro)', 'FCI\\n(exact)']
energies = [hf.e_tot, casci_e, fci_e]
colors = ['#f85149', '#d29922', '#58a6ff']

if nevpt2_e is not None:
    methods.insert(2, 'CASCI +\\nNEVPT2')
    energies.insert(2, nevpt2_e)
    colors.insert(2, '#3fb950')

errors = [abs(e - fci_e) * 1000 for e in energies]

print(f"{'Method':<22s}  {'Energy (Ha)':>14s}  {'Δ FCI (mHa)':>12s}")
print('─' * 52)
for m, e, err in zip(methods, energies, errors):
    label = m.replace('\\n', ' ')
    print(f"{label:<22s}  {e:+14.8f}  {err:10.2f}")""")

code("""# ── Energy Ladder Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                gridspec_kw={'width_ratios': [3, 2]})

# Left: Error bars
bars = ax1.barh(methods, errors, color=colors, edgecolor='#30363d', height=0.6)
ax1.set_xlabel('Error to FCI (mHa)', fontsize=13)
ax1.set_title('Energy Error by Method', fontsize=15, fontweight='bold', pad=15)
ax1.invert_yaxis()
ax1.set_xlim(0, max(errors) * 1.15)

for bar, err in zip(bars, errors):
    if err > 0.01:
        ax1.text(bar.get_width() + max(errors) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f'{err:.2f} mHa', va='center', fontsize=11, color='#c9d1d9')
    else:
        ax1.text(max(errors) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 'exact', va='center', fontsize=11, color='#58a6ff', fontweight='bold')

ax1.axvline(x=1.6, color='#f0883e', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.text(1.6, -0.45, ' chemical\\n accuracy', fontsize=9, color='#f0883e', va='top')

# Right: Absolute energies
y_pos = range(len(methods))
ax2.scatter(energies, y_pos, c=colors, s=120, zorder=5,
            edgecolors='#c9d1d9', linewidth=1)
ax2.set_xlabel('Total Energy (Ha)', fontsize=13)
ax2.set_title('Absolute Energies', fontsize=15, fontweight='bold', pad=15)
ax2.set_yticks(list(y_pos))
ax2.set_yticklabels(methods)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)
ax2.axvline(x=fci_e, color='#58a6ff', linestyle=':', linewidth=1, alpha=0.5)

for i, e in enumerate(energies):
    ax2.text(e, i - 0.3, f'{e:.4f}', fontsize=9, ha='center', color='#8b949e')

plt.tight_layout()
plt.savefig('energy_ladder.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: energy_ladder.png')""")

code("""# ── Natural Orbital Occupations ──
fig, ax = plt.subplots(figsize=(8, 5))

orb_labels = [f'MO {i+1}' for i in range(len(occ))]
bar_colors = ['#3fb950' if n > 0.1 else '#8b949e' for n in occ]
bars = ax.bar(orb_labels, occ, color=bar_colors, edgecolor='#30363d', width=0.6)

ax.set_ylabel('Occupation Number', fontsize=13)
ax.set_title('Natural Orbital Occupations (BeH₂)', fontsize=15,
             fontweight='bold', pad=15)
ax.set_ylim(0, 2.15)
ax.axhline(y=2.0, color='#8b949e', linestyle=':', alpha=0.3, label='Doubly occupied')
ax.axhline(y=0.0, color='#8b949e', linestyle=':', alpha=0.3)

for bar, n in zip(bars, occ):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{n:.4f}', ha='center', va='bottom', fontsize=11, color='#c9d1d9')

ax.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig('natural_occupations.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: natural_occupations.png')""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("full_workflow.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Created full_workflow.ipynb")
