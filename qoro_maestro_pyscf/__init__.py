# This file is part of qoro-maestro-pyscf.
#
# Copyright (C) 2024 Qoro Quantum GmbH
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
qoro-maestro-pyscf
==================
PySCF integration plugin for the Maestro quantum simulator (Qoro Quantum).

Drop-in replacement for ``qiskit-nature-pyscf``, using Maestro's native
GPU-accelerated backends instead of Qiskit.

Primary API
-----------
.. autosummary::
    MaestroSolver
    BackendConfig
    configure_backend

Quick Start
-----------
::

    from pyscf import gto, scf, mcscf
    from qoro_maestro_pyscf import MaestroSolver

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    hf  = scf.RHF(mol).run()

    cas = mcscf.CASCI(hf, 2, 2)
    cas.fcisolver = MaestroSolver(ansatz="uccsd")
    cas.run()
"""

from qoro_maestro_pyscf.maestro_solver import MaestroSolver
from qoro_maestro_pyscf.backends import BackendConfig, configure_backend, set_license_key
from qoro_maestro_pyscf.properties import (
    compute_dipole_moment,
    compute_natural_orbitals,
)

__all__ = [
    "MaestroSolver",
    "BackendConfig",
    "configure_backend",
    "set_license_key",
    "compute_dipole_moment",
    "compute_natural_orbitals",
]
__version__ = "0.1.0"
