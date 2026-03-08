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
Expectation value engine using Maestro's native QuantumCircuit.estimate().

Wraps circuit evaluation so that the rest of the library can call a single
function without worrying about backend configuration details.
"""

from __future__ import annotations

import numpy as np

from maestro.circuits import QuantumCircuit

from qoro_maestro_pyscf.backends import BackendConfig


def evaluate_expectation(
    circuit: QuantumCircuit,
    pauli_labels: list[str],
    config: BackendConfig,
) -> np.ndarray:
    """
    Evaluate expectation values of Pauli observables on a Maestro circuit.

    All observables are batched into a single ``qc.estimate()`` call, so
    Maestro evaluates them in one statevector (or MPS) pass on the GPU.

    Parameters
    ----------
    circuit : QuantumCircuit
        The prepared (parameterised) circuit.
    pauli_labels : list[str]
        Pauli observable strings, e.g. ``["ZZII", "IXYZ"]``.
    config : BackendConfig
        Maestro backend configuration.

    Returns
    -------
    expectation_values : np.ndarray, shape (len(pauli_labels),)
        Real-valued expectation values ⟨ψ|Pᵢ|ψ⟩.
    """
    if not pauli_labels:
        return np.array([], dtype=float)

    result = circuit.estimate(
        observables=pauli_labels,
        simulator_type=config.simulator_type,
        simulation_type=config.simulation_type,
    )

    return np.array(result["expectation_values"], dtype=float)


def compute_energy(
    circuit: QuantumCircuit,
    identity_offset: float,
    pauli_labels: list[str],
    pauli_coeffs: np.ndarray,
    config: BackendConfig,
) -> float:
    """
    Compute the total energy ⟨H⟩ = c₀ + Σᵢ Re(cᵢ)·⟨Pᵢ⟩ for a given circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The prepared circuit.
    identity_offset : float
        Coefficient of the identity term.
    pauli_labels : list[str]
        Non-identity Pauli terms.
    pauli_coeffs : np.ndarray
        Complex coefficients for each Pauli term.
    config : BackendConfig
        Maestro backend configuration.

    Returns
    -------
    energy : float
        The expectation value of the Hamiltonian.
    """
    exp_vals = evaluate_expectation(circuit, pauli_labels, config)
    # Coefficients are real for a Hermitian Hamiltonian (physical observable).
    # We use .real to drop any floating-point imaginary noise from OpenFermion.
    return identity_offset + float(np.dot(pauli_coeffs.real, exp_vals))
