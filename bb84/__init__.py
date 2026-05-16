"""
bb84
====
BB84 Quantum Key Distribution protocol implementation.

Submodules
──────────
  config             — SimulationConfig, SimulationResult, QBERResult dataclasses
  core               — Alice, Bob, Eve, key sifting, QBER estimation
  noise              — QuantumChannel with depolarizing / T1 / T2 / fibre-loss models
  runner             — Full Qiskit-backed run_simulation() and PRESET_SCENARIOS
  fast               — NumPy density-matrix simulation (~200× faster)
  plots              — Standard analysis plots
  plots_extended     — Research / extended plots
  phase3_plots       — Phase 3 noise model plots
  runner_phase1_ext  — Phase 1 extended runner
"""
