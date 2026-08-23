"""Tests for Zero-Noise Extrapolation integrated as an interactive,
single-config sweep (bb84_zne.run_zne_analysis).

Covers: the extrapolated zero-noise QBER estimate versus the raw
f_scale=1.0 measurement on a genuinely noisy channel, the ValueError
guard for noise models ZNE doesn't apply to, and the degenerate
single-point f_scales case (both extrapolation fallbacks already
implemented in linear_extrapolate/exponential_extrapolate).
"""

from bb84_config import SimulationConfig
from bb84_noise import NoiseModelType
from bb84_zne import run_zne_analysis


def test_extrapolated_estimate_is_lower_than_raw_qber_at_f1():
    base = SimulationConfig(
        n_qubits=1200, seed=10, noise_model=NoiseModelType.DEPOLARIZING,
        depolar_prob=0.05,
    )
    result = run_zne_analysis(base, f_scales=[1.0, 1.5, 2.0, 2.5, 3.0], n_seeds=5)

    assert result.linear_intercept < result.qber_at_f1


def test_fibre_loss_is_not_supported_by_zne():
    base = SimulationConfig(
        n_qubits=600, seed=10, noise_model=NoiseModelType.FIBRE_LOSS,
        channel_length_km=50,
    )
    try:
        run_zne_analysis(base, f_scales=[1.0, 2.0], n_seeds=2)
        assert False, "expected ValueError for fibre_loss"
    except ValueError:
        pass


def test_ideal_channel_is_not_supported_by_zne():
    base = SimulationConfig(n_qubits=600, seed=10)  # noise_model=None -> ideal path
    try:
        run_zne_analysis(base, f_scales=[1.0, 2.0], n_seeds=2)
        assert False, "expected ValueError for an unscalable/unset noise model"
    except ValueError:
        pass


def test_single_f_scale_degenerates_to_existing_fallback_behaviour():
    """f_scales=[1.0] is underdetermined for a slope/fit - this locks in
    the pre-existing fallback behaviour of linear_extrapolate (slope=0,
    weighted mean) and exponential_extrapolate (converged=False) at the
    new integration boundary, rather than introducing new logic."""
    base = SimulationConfig(
        n_qubits=1200, seed=10, noise_model=NoiseModelType.DEPOLARIZING,
        depolar_prob=0.05,
    )
    result = run_zne_analysis(base, f_scales=[1.0], n_seeds=3)

    assert result.linear_slope == 0.0
    assert result.exponential["converged"] is False


def test_zne_reuses_the_base_config_noise_parameters():
    """The sweep should scale from base_config's own noise parameters,
    not hardcoded ones - a distinct depolar_prob baseline should shift
    the qber_at_f1 measurement accordingly."""
    low = SimulationConfig(n_qubits=1500, seed=20, noise_model=NoiseModelType.DEPOLARIZING,
                            depolar_prob=0.01)
    high = SimulationConfig(n_qubits=1500, seed=20, noise_model=NoiseModelType.DEPOLARIZING,
                             depolar_prob=0.08)

    r_low = run_zne_analysis(low, f_scales=[1.0, 2.0], n_seeds=5)
    r_high = run_zne_analysis(high, f_scales=[1.0, 2.0], n_seeds=5)

    assert r_high.qber_at_f1 > r_low.qber_at_f1
