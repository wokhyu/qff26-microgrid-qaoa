"""IBM Quantum execution: backends, error mitigation options, job logging.

Nothing here needs a token unless you ask for a real backend. Local noisy
simulation uses a fake Heron backend from ``qiskit_ibm_runtime.fake_provider``,
which carries a snapshot of a real device's calibration data.

Token setup (once per machine, never commit the token):

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                      token="<API key>", instance="<CRN>")
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

DEFAULT_JOB_LOG = Path(__file__).resolve().parents[2] / "results" / "hardware_jobs.jsonl"


def fake_backend(name: str = "FakeFez"):
    """A fake Heron device for local noisy simulation (FakeFez, FakeTorino, FakeMarrakesh...)."""
    from qiskit_ibm_runtime import fake_provider

    return getattr(fake_provider, name)()


def noisy_simulator(name: str = "FakeFez", seed: int = 7):
    """AerSimulator carrying the fake device's noise model, coupling map and basis gates.

    Build it once and reuse it: constructing the noise model takes seconds.
    Pass it as ``mode=`` to ``qiskit_ibm_runtime.SamplerV2`` for local runs.
    """
    from qiskit_aer import AerSimulator

    sim = AerSimulator.from_backend(fake_backend(name))
    sim.set_options(seed_simulator=seed)
    return sim


def real_backend(name: str | None = None, min_qubits: int = 12):
    """Least busy real device, or a named one. Requires a saved account."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    if name:
        return service.backend(name)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)


def transpile_for(circuit: QuantumCircuit, backend, optimization_level: int = 3, seed: int = 7) -> QuantumCircuit:
    """ISA circuit for the backend's native gates and coupling map."""
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level, backend=backend, seed_transpiler=seed
    )
    return pm.run(circuit)


def mitigated_sampler(mode, shots: int = 4096, dynamical_decoupling: bool = True, twirling: bool = True):
    """SamplerV2 with dynamical decoupling and Pauli twirling.

    ``mode`` is a backend, a ``Batch`` or a fake backend (local simulation;
    mitigation options are ignored there).
    """
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=mode)
    sampler.options.default_shots = shots
    if dynamical_decoupling:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    if twirling:
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        sampler.options.twirling.num_randomizations = "auto"
    return sampler


def zne_estimator(mode, noise_factors=(1, 3, 5), dynamical_decoupling: bool = True):
    """EstimatorV2 with zero-noise extrapolation, TREX readout mitigation and twirling."""
    from qiskit_ibm_runtime import EstimatorV2

    estimator = EstimatorV2(mode=mode)
    estimator.options.resilience_level = 2
    estimator.options.resilience.zne_mitigation = True
    estimator.options.resilience.zne.noise_factors = tuple(noise_factors)
    estimator.options.resilience.zne.extrapolator = ("exponential", "linear")
    estimator.options.resilience.measure_mitigation = True
    estimator.options.twirling.enable_gates = True
    if dynamical_decoupling:
        estimator.options.dynamical_decoupling.enable = True
        estimator.options.dynamical_decoupling.sequence_type = "XpXm"
    return estimator


def log_job(job, tag: str, path: Path = DEFAULT_JOB_LOG, **extra) -> dict:
    """Append the job ID to results/hardware_jobs.jsonl so judges can verify the run."""
    try:
        backend_name = job.backend().name
    except Exception:  # local jobs may not expose a backend object
        backend_name = "local"
    record = {
        "job_id": job.job_id(),
        "backend": backend_name,
        "tag": tag,
        "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record
