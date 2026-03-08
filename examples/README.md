# Examples

Hands-on examples demonstrating `qoro-maestro-pyscf` for quantum chemistry on Maestro's GPU-accelerated simulator.

## Quick Start

```bash
pip install qoro-maestro-pyscf
cd examples/

# Run any example
python 01_h2_dissociation.py
python 07_nevpt2.py --gpu
python 10_full_workflow.py --gpu
```

## Examples

### Getting Started

| # | Script | Topic | Key Concepts |
|---|--------|-------|-------------|
| 1 | [01_h2_dissociation.py](01_h2_dissociation.py) | H₂ bond-breaking curve | PES scan, multi-reference character, VQE vs FCI |
| 2 | [02_lih_uccsd.py](02_lih_uccsd.py) | LiH with UCCSD ansatz | Chemistry-motivated ansatz, CASCI |
| 3 | [03_gpu_vs_cpu.py](03_gpu_vs_cpu.py) | GPU acceleration benchmark | Backend switching, timing, Maestro speedup |

### Simulation Backends

| # | Script | Topic | Key Concepts |
|---|--------|-------|-------------|
| 4 | [04_mps_bond_dimension.py](04_mps_bond_dimension.py) | MPS GPU simulation | Bond dimension trade-offs, H₂O |
| 5 | [05_statevector_vs_mps.py](05_statevector_vs_mps.py) | Simulation modes | Accuracy vs performance comparison |

### Production Chemistry

| # | Script | Topic | Key Concepts |
|---|--------|-------|-------------|
| 6 | [06_casscf_orbital_opt.py](06_casscf_orbital_opt.py) | CASSCF | Orbital optimisation, RDMs, CASCI vs CASSCF |
| 7 | [07_nevpt2.py](07_nevpt2.py) | Perturbation theory | Dynamic correlation, NEVPT2 on top of CASSCF |
| 8 | [08_dipole_moment.py](08_dipole_moment.py) | Molecular properties | Dipole moments, natural orbital occupations |
| 9 | [09_geometry_opt.py](09_geometry_opt.py) | Geometry optimisation | Equilibrium bond length, PES scan |
| 10 | [full_workflow/](full_workflow/) | Complete pipeline | BeH₂: HF → CASSCF → NEVPT2 → properties |

## GPU vs CPU

All examples support `--gpu` and `--cpu` flags:

```bash
python 07_nevpt2.py --gpu    # NVIDIA GPU (requires license)
python 07_nevpt2.py --cpu    # CPU fallback (no GPU needed)
```

## License Key

For GPU examples, set your Maestro license key:

```bash
export MAESTRO_LICENSE_KEY="XXXX-XXXX-XXXX-XXXX"
```
