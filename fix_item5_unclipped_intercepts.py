"""
fix_item5_unclipped_intercepts.py
==================================
Addresses mentor Item 5 (Critical):

  For p_eve=0 (no-eavesdropper), the fitted ZNE linear intercept is slightly
  negative (confirmed by Fig. 1). The existing code clips it with max(0, a),
  making the zero-intercept look exact BY CONSTRUCTION — zero bias from a
  mathematical artifact, not from the estimator actually working perfectly.

  This script:
    1. Runs the ZNE sweep at all tested p_eve values (including p_eve=0)
       with the 40-seed budget.
    2. Reports BOTH the clipped (max(0,a)) and unclipped (raw a) intercepts
       in a single table.
    3. Recomputes the bias metrics on the UNCLIPPED intercepts so the
       no-false-alarm claim is honest.
    4. Prints suggested replacement text for Table 6 / the control section.

USAGE
-----
  Option A — use cached results:
      python fix_item5_unclipped_intercepts.py --from-cache

  Option B — recompute (slow):
      python fix_item5_unclipped_intercepts.py --recompute [--seeds 40]

  Place in the same directory as bb84_zne.py, bb84_runner.py, etc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── configuration ────────────────────────────────────────────────────
RESULTS_JSON      = "results_summary.json"
P_EVE_GRID        = [0.0, 0.1, 0.2, 0.3, 0.4]
F_SCALES          = [1.0, 1.5, 2.0, 2.5]
DPOLAR_LEVELS     = [0.010, 0.030, 0.050, 0.080]  # control table depolar levels
N_QUBITS          = 2000
BASE_DEPOLAR_PROB = 0.03
N_SEEDS_DEFAULT   = 40
SEED_LIST         = list(range(N_SEEDS_DEFAULT))
PASS_TOLERANCE_PP = 2.0    # same tolerance as Table 6 / tab:control in paper


# ── load cached raw results ──────────────────────────────────────────

def load_cached(path: str) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    for key in ("headline_bias_reduction_hq", "zne_sweep_40seed", "zne_sweep"):
        if key in data:
            return data[key].get("raw", data[key])
    return None


# ── recompute raw sweep ───────────────────────────────────────────────

def recompute(seeds: List[int], dpolar: float = BASE_DEPOLAR_PROB) -> Dict:
    """Run a fresh ZNE sweep and return raw[p_eve][f_scale] = [(q%, lo%, hi%),...]"""
    try:
        from bb84_zne import run_zne_sweep
        from bb84_noise import NoiseModelType
    except ImportError:
        print("ERROR: cannot import bb84_zne or bb84_noise. Run from project dir.")
        sys.exit(1)

    result = run_zne_sweep(
        base_noise_model=NoiseModelType.DEPOLARIZING,
        f_scales=F_SCALES,
        p_eve_grid=P_EVE_GRID,
        seeds=seeds,
        n_qubits=N_QUBITS,
        base_depolar_prob=dpolar,
    )
    return result["raw"]


def recompute_control_levels(seeds: List[int]) -> Dict[float, Dict]:
    """
    Run ZNE sweep at p_eve=0 for each depolarizing level in DPOLAR_LEVELS.
    Returns {dpolar: raw[p_eve=0][f_scale]=[...]}
    """
    from bb84_zne import run_zne_sweep
    from bb84_noise import NoiseModelType

    out = {}
    for dp in DPOLAR_LEVELS:
        print(f"  Computing control: depolar={dp} … ", end="", flush=True)
        result = run_zne_sweep(
            base_noise_model=NoiseModelType.DEPOLARIZING,
            f_scales=F_SCALES,
            p_eve_grid=[0.0],
            seeds=seeds,
            n_qubits=N_QUBITS,
            base_depolar_prob=dp,
        )
        out[dp] = result["raw"]
        print("done")
    return out


# ── core: extract clipped AND unclipped intercepts ───────────────────

def extract_intercepts(
    raw: Dict,
    seeds: List[int],
) -> Dict[float, Dict]:
    """
    For each p_eve, fit weighted linear ZNE and return both:
      - a_raw     : unclipped intercept (the raw regression coefficient)
      - a_clipped : max(0, a_raw)  [what the current code reports]
      - true_qber : p_eve / 4 * 100  (true Eve-only contribution in %)
      - bias_clipped : |a_clipped - true_qber|
      - bias_unclipped : |a_raw - true_qber|

    Each is computed over the full seed set (mean of per-seed intercepts
    is NOT the same as fitting to the seed-mean — we fit to the mean
    QBER across seeds at each f, which is what the paper does).
    """
    try:
        from bb84_zne import linear_extrapolate
    except ImportError:
        print("ERROR: cannot import bb84_zne. Run from project directory.")
        sys.exit(1)

    results = {}
    for p_eve in P_EVE_GRID:
        if p_eve not in raw:
            continue
        true_q = p_eve / 4 * 100

        # Aggregate across seeds (exactly as the paper does)
        f_vals, q_means, weights = [], [], []
        for f in F_SCALES:
            if f not in raw[p_eve]:
                continue
            vals = raw[p_eve][f][:len(seeds)]
            qbers     = [v[0] for v in vals]
            halfwidths= [max(1e-3, (v[2]-v[1])/2) for v in vals]
            mean_q = float(np.mean(qbers))
            mean_hw = float(np.mean(halfwidths))
            f_vals.append(f)
            q_means.append(mean_q)
            weights.append(1.0 / mean_hw**2)

        if len(f_vals) < 2:
            continue

        a_raw, slope = linear_extrapolate(f_vals, q_means, weights)
        a_clipped = max(0.0, a_raw)

        # Per-seed intercepts (for bootstrap CI on unclipped)
        per_seed_intercepts_raw = []
        for s_idx in range(len(seeds)):
            fv, qv, wv = [], [], []
            for f in F_SCALES:
                if f not in raw[p_eve] or s_idx >= len(raw[p_eve][f]):
                    continue
                qb, ci_lo, ci_hi = raw[p_eve][f][s_idx]
                hw = max(1e-3, (ci_hi-ci_lo)/2)
                fv.append(f); qv.append(qb); wv.append(1/hw**2)
            if len(fv) >= 2:
                ai, _ = linear_extrapolate(fv, qv, wv)
                per_seed_intercepts_raw.append(ai)

        # Bootstrap CI on unclipped intercepts
        if len(per_seed_intercepts_raw) >= 5:
            rng = np.random.default_rng(42)
            boot_ints = []
            for _ in range(2000):
                sample = rng.choice(per_seed_intercepts_raw, size=len(per_seed_intercepts_raw), replace=True)
                boot_ints.append(float(np.mean(sample)))
            ci_lo_unc = float(np.percentile(boot_ints, 2.5))
            ci_hi_unc = float(np.percentile(boot_ints, 97.5))
        else:
            ci_lo_unc = ci_hi_unc = float("nan")

        results[p_eve] = dict(
            true_q=true_q,
            a_raw=a_raw,
            a_clipped=a_clipped,
            bias_clipped=abs(a_clipped - true_q),
            bias_unclipped=abs(a_raw - true_q),
            ci_lo_unclipped=ci_lo_unc,
            ci_hi_unclipped=ci_hi_unc,
        )
    return results


def extract_control_table(
    control_raw: Dict[float, Dict],
    seeds: List[int],
) -> List[Dict]:
    """
    Re-compute the no-eavesdropper control table (tab:control in paper)
    with both clipped and unclipped ZNE intercepts.
    """
    from bb84_zne import linear_extrapolate

    rows = []
    for dp in DPOLAR_LEVELS:
        raw = control_raw.get(dp, {})
        if 0.0 not in raw:
            continue
        # raw QBER at f=1 (mean over seeds)
        if 1.0 in raw[0.0]:
            raw_f1 = float(np.mean([v[0] for v in raw[0.0][1.0][:len(seeds)]]))
        else:
            raw_f1 = float("nan")

        # ZNE intercept
        f_vals, q_means, weights_list = [], [], []
        for f in F_SCALES:
            if f not in raw[0.0]: continue
            vals = raw[0.0][f][:len(seeds)]
            qbers = [v[0] for v in vals]
            hws   = [max(1e-3, (v[2]-v[1])/2) for v in vals]
            f_vals.append(f)
            q_means.append(float(np.mean(qbers)))
            weights_list.append(1.0 / float(np.mean(hws))**2)

        if len(f_vals) < 2:
            continue
        a_raw, _ = linear_extrapolate(f_vals, q_means, weights_list)
        a_clipped = max(0.0, a_raw)

        # Bootstrap CI on unclipped
        per_seed = []
        for s_idx in range(len(seeds)):
            fv, qv, wv = [], [], []
            for f in F_SCALES:
                if f not in raw[0.0] or s_idx >= len(raw[0.0][f]): continue
                qb, ci_lo, ci_hi = raw[0.0][f][s_idx]
                hw = max(1e-3, (ci_hi-ci_lo)/2)
                fv.append(f); qv.append(qb); wv.append(1/hw**2)
            if len(fv) >= 2:
                ai, _ = linear_extrapolate(fv, qv, wv)
                per_seed.append(ai)

        if len(per_seed) >= 5:
            rng = np.random.default_rng(42)
            boot = [float(np.mean(rng.choice(per_seed, size=len(per_seed), replace=True)))
                    for _ in range(2000)]
            ci_lo_unc = float(np.percentile(boot, 2.5))
            ci_hi_unc = float(np.percentile(boot, 97.5))
        else:
            ci_lo_unc = ci_hi_unc = float("nan")

        rows.append(dict(
            depolar=dp,
            raw_f1=raw_f1,
            zne_clipped=a_clipped,
            zne_unclipped=a_raw,
            ci_lo=ci_lo_unc,
            ci_hi=ci_hi_unc,
            # Pass with 2pp tolerance using UNCLIPPED value
            pass_unc=(abs(a_raw) <= PASS_TOLERANCE_PP and ci_hi_unc <= PASS_TOLERANCE_PP),
        ))
    return rows


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache",  action="store_true")
    parser.add_argument("--recompute",   action="store_true")
    parser.add_argument("--cache-path",  default=RESULTS_JSON)
    parser.add_argument("--seeds",       type=int, default=N_SEEDS_DEFAULT)
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    # ── load or compute main sweep raw data ─────────────────────────
    raw_main = None
    if args.from_cache:
        raw_main = load_cached(args.cache_path)
        if raw_main is None:
            print("[warn] Cache missing or key not found — recomputing.")

    if raw_main is None:
        print(f"Recomputing ZNE sweep ({len(seeds)} seeds) …")
        raw_main = recompute(seeds)

    # ── extract clipped vs unclipped intercepts ──────────────────────
    intercepts = extract_intercepts(raw_main, seeds)

    print("\n" + "=" * 75)
    print("ITEM 5 FIX: Clipped vs Unclipped ZNE Intercepts")
    print("=" * 75)

    print(f"\n{'p_eve':>6}  {'True%':>7}  "
          f"{'Raw(f=1)%':>10}  "
          f"{'ZNE clipped':>12}  {'ZNE unclipped':>14}  "
          f"{'Bias clipped':>13}  {'Bias unclipped':>15}  "
          f"{'95% CI (unclipped)':>20}")
    print("-" * 110)

    # Also grab mean raw QBER at f=1
    for p_eve in P_EVE_GRID:
        if p_eve not in intercepts: continue
        r = intercepts[p_eve]
        raw_f1 = float("nan")
        if p_eve in raw_main and 1.0 in raw_main[p_eve]:
            raw_f1 = float(np.mean([v[0] for v in raw_main[p_eve][1.0][:len(seeds)]]))

        ci_str = (f"[{r['ci_lo_unclipped']:+.2f}, {r['ci_hi_unclipped']:+.2f}]"
                  if not np.isnan(r['ci_lo_unclipped']) else "  [n/a]")
        print(f"{p_eve:>6.1f}  {r['true_q']:>7.2f}%  "
              f"{raw_f1:>10.2f}%  "
              f"{r['a_clipped']:>12.2f}%  {r['a_raw']:>14.2f}%  "
              f"{r['bias_clipped']:>13.2f} pp  {r['bias_unclipped']:>15.2f} pp  "
              f"{ci_str:>20}")

    print("\n*** KEY FINDING: at p_eve=0 the raw intercept is "
          f"{intercepts.get(0.0, {}).get('a_raw', float('nan')):.3f}% "
          "(slightly negative).")
    print("    max(0, a) clips this to 0.00%, making bias look ZERO by construction.")
    print("    Report BOTH columns. Use 'unclipped' for all bias metrics.")

    # ── aggregate bias comparison ────────────────────────────────────
    biases_clipped   = [intercepts[p]["bias_clipped"]   for p in P_EVE_GRID if p in intercepts]
    biases_unclipped = [intercepts[p]["bias_unclipped"] for p in P_EVE_GRID if p in intercepts]
    print(f"\nMean absolute bias (clipped)  : {np.mean(biases_clipped):.3f} pp")
    print(f"Mean absolute bias (unclipped): {np.mean(biases_unclipped):.3f} pp")
    print(f"  NOTE: use the UNCLIPPED number in the paper's bias-reduction ratio.")

    # ── no-eavesdropper control table (per depolar level) ────────────
    print("\n── Re-computing no-eavesdropper control table per depolar level ──")
    if args.recompute:
        control_raw = recompute_control_levels(seeds)
    else:
        # Try to extract from cache — fall back to recompute if missing
        control_raw = {}
        for dp in DPOLAR_LEVELS:
            # Try to find it in the main cache for p_eve=0
            ctrl = recompute([0], dp) if not raw_main else {}
            # For now, use main sweep data at p_eve=0 for dp=0.03 if available
            if dp == BASE_DEPOLAR_PROB and 0.0 in raw_main:
                ctrl = {0.0: raw_main[0.0]}
                control_raw[dp] = ctrl
        if len(control_raw) < len(DPOLAR_LEVELS):
            print("  (Not all depolar levels available from cache — "
                  "run with --recompute for full control table)")

    if control_raw:
        ctrl_rows = extract_control_table(control_raw, seeds)
        if ctrl_rows:
            print(f"\n{'p_dep':>7}  {'Raw(f=1)%':>10}  "
                  f"{'ZNE clipped':>12}  {'ZNE unclipped':>14}  "
                  f"{'95% CI (unclipped)':>22}  {'Pass (unclipped)?':>18}")
            print("-" * 100)
            for row in ctrl_rows:
                ci_str = (f"[{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}]"
                          if not np.isnan(row['ci_lo']) else "[n/a]")
                pass_str = "Yes" if row.get("pass_unc") else "No"
                print(f"{row['depolar']:>7.3f}  {row['raw_f1']:>10.2f}%  "
                      f"{row['zne_clipped']:>12.2f}%  {row['zne_unclipped']:>14.2f}%  "
                      f"{ci_str:>22}  {pass_str:>18}")

    # ── suggested paper text ─────────────────────────────────────────
    print("""
SUGGESTED REVISION for Section 3.4 / Table 6 caption:

  "Table X shows both the clipped estimate [max(0, a)] and the raw fitted
  intercept a. At p_eve=0, the raw intercept is slightly negative
  (a ≈ [INSERT VALUE] pp), which max(0,a) rounds to 0.00% by construction.
  All bias metrics are therefore reported on the unclipped intercept a,
  so that the no-false-alarm claim does not benefit from the floor at zero."
""")

    print("Done. Update Table 6 / tab:control to include an 'Unclipped' column.")


if __name__ == "__main__":
    main()
