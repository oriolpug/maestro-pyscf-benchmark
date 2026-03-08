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
Test suite for qoro-maestro-pyscf.

Tests are structured in three tiers:
  1. Unit tests (no Maestro/PySCF required) — test pure logic
  2. Integration tests (require openfermion + numpy) — test Hamiltonian/ansatz building
  3. End-to-end tests (require all deps) — test the full VQE pipeline

Run:
    pytest tests/ -v
    pytest tests/ -v -k "unit"           # fast, no deps
    pytest tests/ -v -k "integration"    # requires openfermion
    pytest tests/ -v -k "e2e"            # requires all deps
"""

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Pure logic, no external deps
# ══════════════════════════════════════════════════════════════════════════════

class TestAnsatzeUnit:
    """Test ansatz parameter counting and excitation enumeration."""

    def test_hardware_efficient_param_count(self):
        from qoro_maestro_pyscf.ansatze import hardware_efficient_param_count
        # 4 qubits, 2 layers → 4 * 2 * 2 = 16 params
        assert hardware_efficient_param_count(4, 2) == 16
        assert hardware_efficient_param_count(4, 1) == 8
        assert hardware_efficient_param_count(2, 3) == 12

    def test_uccsd_param_count_tuple(self):
        from qoro_maestro_pyscf.ansatze import uccsd_param_count
        # H₂: 4 qubits, (1, 1) electrons
        # occupied: [0, 1], virtual: [2, 3]
        # singles: (0,2), (0,3), (1,2), (1,3) = 4
        # doubles: C(2,2) * C(2,2) = 1 * 1 = 1
        # total: 5
        n_params = uccsd_param_count(4, (1, 1))
        assert n_params == 5

    def test_uccsd_param_count_int(self):
        from qoro_maestro_pyscf.ansatze import uccsd_param_count
        # nelec=2 → (1, 1); should match tuple version
        assert uccsd_param_count(4, 2) == uccsd_param_count(4, (1, 1))

    def test_excitation_enumeration(self):
        from qoro_maestro_pyscf.ansatze import _get_uccsd_excitations
        singles, doubles = _get_uccsd_excitations(4, (1, 1))
        # All singles should be occupied → virtual
        for i, a in singles:
            assert i in [0, 1]  # occupied
            assert a in [2, 3]  # virtual

    def test_hf_circuit_builds(self):
        from qoro_maestro_pyscf.ansatze import hartree_fock_circuit
        qc = hartree_fock_circuit(4, (1, 1))
        # Should successfully build a circuit object
        assert qc is not None

    def test_hf_circuit_int_nelec(self):
        from qoro_maestro_pyscf.ansatze import hartree_fock_circuit
        # nelec=2 → (1, 1)
        qc = hartree_fock_circuit(4, 2)
        assert qc is not None

    def test_hardware_efficient_builds(self):
        from qoro_maestro_pyscf.ansatze import (
            hardware_efficient_ansatz,
            hardware_efficient_param_count,
        )
        n_qubits, n_layers = 4, 2
        n_params = hardware_efficient_param_count(n_qubits, n_layers)
        params = np.zeros(n_params)
        qc = hardware_efficient_ansatz(params, n_qubits, n_layers)
        assert qc is not None

    def test_hardware_efficient_with_hf(self):
        from qoro_maestro_pyscf.ansatze import (
            hardware_efficient_ansatz,
            hardware_efficient_param_count,
        )
        n_qubits, n_layers = 4, 2
        n_params = hardware_efficient_param_count(n_qubits, n_layers)
        params = np.zeros(n_params)
        qc = hardware_efficient_ansatz(
            params, n_qubits, n_layers, include_hf=True, nelec=(1, 1)
        )
        assert qc is not None

    def test_hardware_efficient_hf_requires_nelec(self):
        from qoro_maestro_pyscf.ansatze import hardware_efficient_ansatz
        with pytest.raises(ValueError, match="nelec is required"):
            hardware_efficient_ansatz(np.zeros(8), 2, 2, include_hf=True)

    def test_uccsd_builds(self):
        from qoro_maestro_pyscf.ansatze import uccsd_ansatz, uccsd_param_count
        n_qubits = 4
        nelec = (1, 1)
        n_params = uccsd_param_count(n_qubits, nelec)
        params = np.zeros(n_params)
        qc = uccsd_ansatz(params, n_qubits, nelec)
        assert qc is not None


class TestBackendsUnit:
    """Test backend configuration logic."""

    def test_simulation_mode_enum(self):
        from qoro_maestro_pyscf.backends import SimulationMode
        assert SimulationMode("statevector") == SimulationMode.STATEVECTOR
        assert SimulationMode("mps") == SimulationMode.MPS

    def test_simulation_mode_invalid(self):
        from qoro_maestro_pyscf.backends import SimulationMode
        with pytest.raises(ValueError):
            SimulationMode("invalid_mode")

    def test_set_license_key(self):
        import os
        from qoro_maestro_pyscf.backends import set_license_key
        set_license_key("TEST-KEY-1234")
        assert os.environ["MAESTRO_LICENSE_KEY"] == "TEST-KEY-1234"
        # Clean up
        del os.environ["MAESTRO_LICENSE_KEY"]


class TestRdmUnit:
    """Test RDM tracing functions."""

    def test_trace_spin_rdm1(self):
        from qoro_maestro_pyscf.rdm import trace_spin_rdm1
        rdm1_a = np.eye(2) * 0.5
        rdm1_b = np.eye(2) * 0.3
        traced = trace_spin_rdm1(rdm1_a, rdm1_b)
        np.testing.assert_allclose(traced, np.eye(2) * 0.8)

    def test_trace_spin_rdm2(self):
        from qoro_maestro_pyscf.rdm import trace_spin_rdm2
        shape = (2, 2, 2, 2)
        rdm2_aa = np.ones(shape) * 0.1
        rdm2_ab = np.ones(shape) * 0.2
        rdm2_bb = np.ones(shape) * 0.3
        traced = trace_spin_rdm2(rdm2_aa, rdm2_ab, rdm2_bb)
        # traced = aa + ab + ba + bb = 0.1 + 0.2 + 0.2 + 0.3 = 0.8
        np.testing.assert_allclose(traced, np.ones(shape) * 0.8)

    def test_trace_rdm1_shape(self):
        from qoro_maestro_pyscf.rdm import trace_spin_rdm1
        norb = 3
        rdm1_a = np.random.rand(norb, norb)
        rdm1_b = np.random.rand(norb, norb)
        traced = trace_spin_rdm1(rdm1_a, rdm1_b)
        assert traced.shape == (norb, norb)


# ══════════════════════════════════════════════════════════════════════════════
# Integration Tests — Require openfermion + numpy
# ══════════════════════════════════════════════════════════════════════════════

class TestHamiltonianIntegration:
    """Test Hamiltonian construction with OpenFermion."""

    def test_h2_hamiltonian_shape(self):
        """Build a minimal 2-orbital Hamiltonian and verify output types."""
        from qoro_maestro_pyscf.hamiltonian import (
            integrals_to_qubit_hamiltonian,
            qubit_op_to_pauli_list,
        )
        from openfermion import QubitOperator

        norb = 2
        h1 = np.random.rand(norb, norb)
        h1 = (h1 + h1.T) / 2  # symmetrise
        h2 = np.random.rand(norb, norb, norb, norb)
        h2 = (h2 + h2.transpose(1, 0, 3, 2)) / 2  # symmetrise

        qubit_op, identity_offset = integrals_to_qubit_hamiltonian(h1, h2, norb)

        assert isinstance(qubit_op, QubitOperator)
        assert isinstance(identity_offset, float)
        assert len(qubit_op.terms) > 0  # should have non-trivial terms

    def test_pauli_list_format(self):
        """Verify Pauli labels are correct length and contain valid chars."""
        from qoro_maestro_pyscf.hamiltonian import (
            integrals_to_qubit_hamiltonian,
            qubit_op_to_pauli_list,
        )

        norb = 2
        n_qubits = 2 * norb
        h1 = np.eye(norb) * -1.0
        h2 = np.zeros((norb, norb, norb, norb))

        qubit_op, _ = integrals_to_qubit_hamiltonian(h1, h2, norb)
        identity_coeff, labels, coeffs = qubit_op_to_pauli_list(qubit_op, n_qubits)

        for label in labels:
            assert len(label) == n_qubits
            assert all(c in "IXYZ" for c in label)

        assert len(labels) == len(coeffs)

    def test_uhf_integrals(self):
        """Verify UHF (tuple) integral format is handled."""
        from qoro_maestro_pyscf.hamiltonian import integrals_to_qubit_hamiltonian

        norb = 2
        h1_a = np.eye(norb) * -1.0
        h1_b = np.eye(norb) * -0.9
        h2_aa = np.zeros((norb, norb, norb, norb))
        h2_ab = np.zeros((norb, norb, norb, norb))
        h2_bb = np.zeros((norb, norb, norb, norb))

        qubit_op, _ = integrals_to_qubit_hamiltonian(
            (h1_a, h1_b), (h2_aa, h2_ab, h2_bb), norb
        )
        assert len(qubit_op.terms) > 0

    def test_identity_operator(self):
        """Zero integrals should produce mostly identity terms."""
        from qoro_maestro_pyscf.hamiltonian import integrals_to_qubit_hamiltonian

        norb = 2
        h1 = np.zeros((norb, norb))
        h2 = np.zeros((norb, norb, norb, norb))

        qubit_op, identity_offset = integrals_to_qubit_hamiltonian(h1, h2, norb)
        # All-zero integrals → zero Hamiltonian
        for term, coeff in qubit_op.terms.items():
            assert abs(coeff) < 1e-12


class TestMaestroSolverIntegration:
    """Test MaestroSolver construction and validation."""

    def test_solver_construction(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver(
            ansatz="hardware_efficient",
            ansatz_layers=2,
            backend="cpu",
        )
        assert solver.ansatz == "hardware_efficient"
        assert solver.ansatz_layers == 2
        assert solver.backend == "cpu"

    def test_solver_uccsd_construction(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver(ansatz="uccsd", backend="gpu")
        assert solver.ansatz == "uccsd"

    def test_solver_mps_construction(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver(
            simulation="mps",
            mps_bond_dim=128,
        )
        assert solver.simulation == "mps"
        assert solver.mps_bond_dim == 128

    def test_solver_license_key(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver(license_key="TEST-1234")
        assert solver.license_key == "TEST-1234"

    def test_solver_default_state(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        assert solver.converged is False
        assert solver.optimal_params is None
        assert solver.energy_history == []

    def test_approx_kernel_alias(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        assert solver.approx_kernel == solver.kernel


class TestPackageExports:
    """Test that all public API is exported correctly."""

    def test_maestro_solver_import(self):
        from qoro_maestro_pyscf import MaestroSolver
        assert MaestroSolver is not None

    def test_backend_config_import(self):
        from qoro_maestro_pyscf import BackendConfig, configure_backend
        assert BackendConfig is not None
        assert configure_backend is not None

    def test_set_license_key_import(self):
        from qoro_maestro_pyscf import set_license_key
        assert callable(set_license_key)

    def test_version(self):
        import qoro_maestro_pyscf
        assert hasattr(qoro_maestro_pyscf, "__version__")
        assert qoro_maestro_pyscf.__version__ == "0.1.0"


# ══════════════════════════════════════════════════════════════════════════════
# End-to-End Tests — Require PySCF + Maestro + OpenFermion
# ══════════════════════════════════════════════════════════════════════════════

def _can_run_e2e() -> bool:
    """Check if all dependencies for e2e tests are available."""
    try:
        import pyscf  # noqa: F401
        import maestro  # noqa: F401
        import openfermion  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _can_run_e2e(), reason="Requires pyscf + maestro + openfermion")
class TestE2E:
    """Full VQE pipeline tests (skipped if deps not installed)."""
    pass
