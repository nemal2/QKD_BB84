"""
fix_item7_randomf_countermeasure.py
=====================================
Tests the RandomF countermeasure against LinearScalingEve (Item 7).

THE ATTACK BEING COUNTERED
---------------------------
LinearScalingEve works because she observes the noise-scaling schedule
{f=1.0, 1.5, 2.0, 2.5} and scales her own interception rate to match:
  p_eve(f) = p_eve_base * f
This makes her contribution linear in f just like hardware noise, so
ZNE-linear cannot separate the two.  Bias grows to ~5 pp (Table adaptive).

THE COUNTERMEASURE
------------------
Alice and Bob choose f randomly and secretly each round, interleaved so
the order is not predictable.  Eve can observe the QBER rising and falling
but cannot reliably infer which f is in use for any given batch.  Her only
safe strategy is to fix her rate at p_eve_base (what she set at f=1) and
not scale it.  From ZNE's perspective this is now equivalent to a static
Eve, and ZNE-linear should recover its bias-reduction guarantee.

WHAT THE SIMULATION DOES
-------------------------
Three conditions, all at p_eve_base = 0.2 (true contribution = 5.00%):

1. STATIC_EVE          - baseline; ZNE should work (bias ~0.18 pp)
2. LINEAR_SCALING_EVE  - known attack; ZNE fails (bias ~5.00 pp)
3. RANDOMF_COUNTER     - LinearScalingEve but Eve only knows f=1, so she
                         cannot scale.  In code: eve_factory always passes
                         f_scale=1.0 to LinearScalingEve regardless of the
                         actual f used for noise.  ZNE should recover.

USAGE
-----
  python fix_item7_randomf_countermeasure.py

  Takes ~5-10 minutes with 10 seeds (enough to confirm the result).
  Increase N_SEEDS to 40 for paper-quality numbers.
"""

from __future__ import annotations

import numpy as np
from bb84_eve_adaptive import (
    LinearScalingEve,
    BasisBiasedEve,
    run_adaptive_zne_sweep,
)
from bb84_core import Eve as StaticEve
from bb84_noise import NoiseModelType

# ── parameters ───────────────────────────────────────────────────────
P_EVE_BASE    = 0.2      # true Eve contribution = p_eve_base/4 * 100 = 5.00%
N_QUBITS      = 2000
BASE_DEPOLAR  = 0.03
F_SCALES      = [1.0, 1.5, 2.0, 2.5]
N_SEEDS       = 10       # increase to 40 for paper; 10 is enough to see the effect
SEED_LIST     = list(range(N_SEEDS))
TRUE_CONTRIB  = P_EVE_BASE / 4 * 100   # 5.00%


# ── three Eve factories ───────────────────────────────────────────────

def factory_static(f_scale, seed):
    """Standard static intercept-resend Eve — ZNE should work."""
    eve = StaticEve(intercept_prob=P_EVE_BASE, seed=seed)
    return eve


def factory_linear_scaling(f_scale, seed):
    """LinearScalingEve who KNOWS f and scales accordingly — ZNE fails."""
    return LinearScalingEve(p_eve_base=P_EVE_BASE, f_scale=f_scale, seed=seed)


def factory_randomf_counter(f_scale, seed):
    """
    LinearScalingEve who does NOT know f.
    Alice/Bob randomize f secretly, so Eve's best strategy is to stay
    at her base rate.  We model this by always passing f_scale=1.0 to
    LinearScalingEve regardless of the actual f being used for noise.
    Her intercept probability stays fixed at p_eve_base * 1.0 = p_eve_base.
    """
    return LinearScalingEve(p_eve_base=P_EVE_BASE, f_scale=1.0, seed=seed)


# ── run all three conditions ──────────────────────────────────────────

def run_condition(name: str, factory) -> dict:
    print(f"  Running: {name} ...", end=" ", flush=True)
    result = run_adaptive_zne_sweep(
        eve_factory        = factory,
        f_scales           = F_SCALES,
        seeds              = SEED_LIST,
        n_qubits           = N_QUBITS,
        base_noise_model   = NoiseModelType.DEPOLARIZING,
        base_depolar_prob  = BASE_DEPOLAR,
    )
    zne_est  = result["zne_linear"]
    raw_f1   = result["qber_at_f1"]
    bias_zne = abs(zne_est - TRUE_CONTRIB)
    bias_raw = abs(raw_f1  - TRUE_CONTRIB)
    print(f"raw={raw_f1:.2f}%  ZNE={zne_est:.2f}%  bias={bias_zne:.2f}pp")
    return dict(name=name, raw=raw_f1, zne=zne_est,
                bias_zne=bias_zne, bias_raw=bias_raw)


def main():
    print("=" * 62)
    print("Item 7: RandomF Countermeasure vs LinearScalingEve")
    print(f"  p_eve_base={P_EVE_BASE}  true contribution={TRUE_CONTRIB:.2f}%")
    print(f"  seeds={N_SEEDS}  N={N_QUBITS}  depolar={BASE_DEPOLAR}")
    print("=" * 62)

    results = []
    results.append(run_condition("Static Eve (baseline)",    factory_static))
    results.append(run_condition("LinearScalingEve (attack)",factory_linear_scaling))
    results.append(run_condition("RandomF countermeasure",   factory_randomf_counter))

    # ── results table ────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    print(f"  True Eve-only contribution: {TRUE_CONTRIB:.2f}%\n")
    print(f"{'Condition':<30} {'Raw@f=1':>8} {'ZNE@f=0':>8} "
          f"{'ZNE bias':>10} {'ZNE works?':>12}")
    print("-" * 72)
    for r in results:
        works = "YES" if r["bias_zne"] < 1.0 else "NO"
        print(f"{r['name']:<30} {r['raw']:>7.2f}%  {r['zne']:>7.2f}%  "
              f"{r['bias_zne']:>8.2f} pp  {works:>12}")

    # ── interpretation ───────────────────────────────────────────────
    counter = results[2]
    linear  = results[1]
    static  = results[0]

    print("\n── Interpretation ──")
    print(f"  Static Eve bias:              {static['bias_zne']:.2f} pp  (expected ~0.18)")
    print(f"  LinearScalingEve bias:        {linear['bias_zne']:.2f} pp  (expected ~5.00)")
    print(f"  RandomF countermeasure bias:  {counter['bias_zne']:.2f} pp")

    if counter["bias_zne"] < 1.5:
        print("\n  ✓ COUNTERMEASURE WORKS: RandomF restores ZNE's bias-reduction")
        print("    guarantee against LinearScalingEve.  Bias is comparable to")
        print("    the static-Eve baseline.")
    else:
        print("\n  ✗ Countermeasure did not fully restore ZNE — investigate further.")

    # ── suggested paper text ─────────────────────────────────────────
    print("""
── Suggested text for Section 3.5 / threat-boundary paragraph ──

"A natural operational countermeasure against LinearScalingEve is to
randomize the noise-scaling schedule: Alice and Bob choose f secretly
and independently each round, interleaved so that the order is not
predictable from the outside.  An adversary who cannot infer the current
f has no basis for scaling her own attack rate and must fix her
interception probability at her chosen base level.  From ZNE-QBER's
perspective this is equivalent to a static Eve.  We confirm this
empirically: when LinearScalingEve is denied knowledge of f (modelled
by fixing her scaling factor to 1.0 regardless of the actual f in use),
ZNE-linear's bias returns to """)
    print(f"    {counter['bias_zne']:.2f} pp, comparable to the static-Eve baseline "
          f"of {static['bias_zne']:.2f} pp and")
    print("""    well within the diagnostic's operating envelope.  Deployments
adopting ZNE-QBER should therefore randomize and conceal the
noise-scaling schedule as a standard operational precaution."
""")


if __name__ == "__main__":
    main()