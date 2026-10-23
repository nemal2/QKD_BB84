#!/usr/bin/env python3
"""
rerun_paper_fixes.py  (v3 — adds Figure 2 and Figure 5 plot generation)
========================================================================
Run this from your project directory (same folder as bb84_zne.py etc.).
Loads cached .nbcache/ results where possible; reruns only what's missing.

Fixes the code-level numerical issues from Dr. Charith's review:
  Fix 1 — Table 4 (tab:zne-main): regenerated from the 40-seed run.
  Fix 2 — Table 3 (tab:stoptests): all three Wilcoxon tests.
  Fix 3 — Clipped vs unclipped intercepts for the no-false-alarm claim.
  Fix 4 — Table 5 (tab:keylength): 40-seed with bootstrap 95% CIs.
  Fix 5 — RQ6 bias is 3.55 pp (text says 5.00 pp).

NEW in v2:
  C2 CHECK — determines whether tab:r1 FER values come from 40 or 60
             trials by finding which denominator gives integer numerators.
  C4 CHECK — verifies Qiskit's phase_damping_error(λ) Kraus convention
             and confirms whether our λ = 1−exp(−t/T₂) formula agrees.

NEW in v3:
  PLOT 1 — Figure 2: ZNE extrapolation plot (no title, paper style)
  PLOT 2 — Figure 5: Key length recovery bar chart (paper style)

Outputs: paper_fix_numbers.json  (paste values from this into LaTeX)
         E11a_zne_extrapolation_hq.png   → replace images/E11a_zne_extrapolation_hq.png
         E11f_key_length_recovery.png    → replace images/E11f_key_length_recovery.png
"""

import hashlib
import math
import json
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import wilcoxon

# ── Parameters — must match bb84_phase5_hardening.ipynb exactly ──────────────
N_QUBITS    = 2000
F_SCALES    = [1.0, 1.5, 2.0, 2.5]
P_EVE_GRID  = [0.0, 0.1, 0.2, 0.3, 0.4]
BASE_DEPOLAR = 0.03
N_SIFTED_FIXED = 993
PASS_TOL_PP    = 2.0

SEEDS_20 = [42,123,7,2024,555,17,99,301,650,777,88,999,321,456,789,111,222,333,444,556]
SEEDS_40 = SEEDS_20 + [1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,
                        1010,1011,1012,1013,1014,1015,1016,1017,1018,1019]
SEEDS_SEC = SEEDS_20[:10]

CACHE_DIR = ".nbcache"


# ═════════════════════════════════════════════════════════════════════════════
# Cache helpers
# ═════════════════════════════════════════════════════════════════════════════

def _tag(n_sh, n_ss):
    return f"scaleTrue_N{N_QUBITS}_sh{n_sh}_ss{n_ss}"

def _cache_path(logical_key: str) -> str:
    h = hashlib.sha1(logical_key.encode()).hexdigest()[:16]
    prefix = logical_key.split("|")[0]
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.pkl")

def load_cache(logical_key: str) -> Optional[object]:
    path = _cache_path(logical_key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_cache(logical_key: str, data: object):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(logical_key), "wb") as f:
        pickle.dump(data, f)


# ═════════════════════════════════════════════════════════════════════════════
# Import project modules
# ═════════════════════════════════════════════════════════════════════════════

def import_project():
    try:
        from bb84_zne import run_zne_sweep, linear_extrapolate, zne_estimate_linear
        from bb84_noise import NoiseModelType
        return run_zne_sweep, linear_extrapolate, zne_estimate_linear, NoiseModelType
    except ImportError as e:
        print(f"ERROR: {e}\nRun this script from the same directory as bb84_zne.py.")
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Load or recompute a headline ZNE sweep
# ═════════════════════════════════════════════════════════════════════════════

def get_raw(seeds, n_secondary, run_zne_sweep, NoiseModelType):
    n_sh = len(seeds)
    key = f"headline_{_tag(n_sh, n_secondary)}"
    cached = load_cache(key)
    if cached is not None:
        print(f"  [cache hit] {key}")
        return cached["raw"] if "raw" in cached else cached
    print(f"  Recomputing ZNE sweep ({n_sh} seeds × {len(F_SCALES)} f-points × "
          f"{len(P_EVE_GRID)} p_eve values = "
          f"{n_sh*len(F_SCALES)*len(P_EVE_GRID)} simulation runs) …")
    result = run_zne_sweep(
        base_noise_model=NoiseModelType.DEPOLARIZING,
        f_scales=F_SCALES, p_eve_grid=P_EVE_GRID, seeds=seeds,
        n_qubits=N_QUBITS, base_depolar_prob=BASE_DEPOLAR,
    )
    save_cache(key, result)
    return result["raw"]


# ═════════════════════════════════════════════════════════════════════════════
# Analysis helpers
# ═════════════════════════════════════════════════════════════════════════════

def intercepts_from_raw(raw, n_seeds, linear_extrapolate):
    out = {}
    for p_eve in P_EVE_GRID:
        if p_eve not in raw:
            continue
        true_q = p_eve * 25.0
        fv, qmeans, wts = [], [], []
        for f in F_SCALES:
            if f not in raw[p_eve]:
                continue
            vals = raw[p_eve][f][:n_seeds]
            qbers = [v[0] for v in vals]
            hws   = [max(1e-3, (v[2]-v[1])/2) for v in vals]
            fv.append(f)
            qmeans.append(float(np.mean(qbers)))
            wts.append(1.0 / float(np.mean(hws))**2)
        if len(fv) < 2:
            continue
        a_raw, _ = linear_extrapolate(fv, qmeans, wts)
        raw_f1 = float(np.mean([v[0] for v in raw[p_eve].get(1.0, [])[:n_seeds]])) \
                 if 1.0 in raw[p_eve] else float("nan")
        out[p_eve] = dict(
            true_q=true_q,
            raw_f1=raw_f1,
            a_raw=a_raw,
            a_clipped=max(0.0, a_raw),
            bias_unclipped=abs(a_raw - true_q),
            bias_clipped=abs(max(0.0, a_raw) - true_q),
        )
    return out


def bootstrap_reduction(raw, seeds, linear_extrapolate, zne_estimate_linear,
                         n_boot=2000, seed=1):
    n = len(seeds)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rb, zb = [], []
        for p_eve in P_EVE_GRID:
            true_q = p_eve * 25.0
            raw_q  = float(np.mean([raw[p_eve][1.0][i][0] for i in idx]))
            fmeans = [float(np.mean([raw[p_eve][f][i][0] for i in idx]))
                      for f in F_SCALES]
            from bb84_zne import linear_extrapolate as _le
            a_raw, _ = _le(F_SCALES, fmeans)
            rb.append(abs(raw_q - true_q))
            zb.append(abs(a_raw - true_q))
        mrb, mzb = np.mean(rb), np.mean(zb)
        if mrb > 0:
            boot.append(100 * (1 - mzb / mrb))
    boot = np.array(boot)
    return float(np.nanmean(boot)), float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def all_three_wilcoxon_tests(raw20, raw40, linear_extrapolate):
    def per_seed_avg_bias(raw, n_seeds):
        rb_list, zb_list = [], []
        for s in range(n_seeds):
            seed_rb, seed_zb = [], []
            for p_eve in P_EVE_GRID:
                true_q = p_eve * 25.0
                if p_eve not in raw or 1.0 not in raw[p_eve]: continue
                if s >= len(raw[p_eve][1.0]): continue
                raw_q = raw[p_eve][1.0][s][0]
                fv, qv, wv = [], [], []
                for f in F_SCALES:
                    if f not in raw[p_eve] or s >= len(raw[p_eve][f]): continue
                    qb, lo, hi = raw[p_eve][f][s]
                    hw = max(1e-3, (hi-lo)/2)
                    fv.append(f); qv.append(qb); wv.append(1/hw**2)
                if len(fv) < 2: continue
                a, _ = linear_extrapolate(fv, qv, wv)
                seed_rb.append(abs(raw_q - true_q))
                seed_zb.append(abs(a - true_q))
            if seed_rb:
                rb_list.append(float(np.mean(seed_rb)))
                zb_list.append(float(np.mean(seed_zb)))
        return np.array(rb_list), np.array(zb_list)

    cells_rb, cells_zb = [], []
    for s in range(len(SEEDS_20)):
        for p_eve in P_EVE_GRID:
            true_q = p_eve * 25.0
            if p_eve not in raw20 or 1.0 not in raw20[p_eve]: continue
            if s >= len(raw20[p_eve][1.0]): continue
            raw_q = raw20[p_eve][1.0][s][0]
            fv, qv, wv = [], [], []
            for f in F_SCALES:
                if f not in raw20[p_eve] or s >= len(raw20[p_eve][f]): continue
                qb, lo, hi = raw20[p_eve][f][s]
                hw = max(1e-3, (hi-lo)/2)
                fv.append(f); qv.append(qb); wv.append(1/hw**2)
            if len(fv) < 2: continue
            a, _ = linear_extrapolate(fv, qv, wv)
            cells_rb.append(abs(raw_q - true_q))
            cells_zb.append(abs(a - true_q))

    stat_bad, p_bad = wilcoxon(cells_rb, cells_zb) if len(cells_rb) >= 10 else (float("nan"), float("nan"))
    rb20, zb20 = per_seed_avg_bias(raw20, len(SEEDS_20))
    stat20, p20 = wilcoxon(rb20, zb20) if len(rb20) >= 10 else (float("nan"), float("nan"))
    rb40, zb40 = per_seed_avg_bias(raw40, len(SEEDS_40))
    stat40, p40 = wilcoxon(rb40, zb40) if len(rb40) >= 10 else (float("nan"), float("nan"))

    return dict(
        invalid_n200=dict(n=len(cells_rb),  stat=float(stat_bad), p=float(p_bad)),
        pilot_n20   =dict(n=int(len(rb20)), stat=float(stat20),   p=float(p20)),
        headline_n40=dict(n=int(len(rb40)), stat=float(stat40),   p=float(p40)),
    )


def bootstrap_key_length(raw, seeds, zne_estimate_linear, n_boot=1000, seed=0):
    def h2(p):
        if p <= 0 or p >= 1: return 0.0
        return -p*math.log2(p) - (1-p)*math.log2(1-p)

    n = len(seeds)
    rng = np.random.default_rng(seed)
    rows = {}
    for p_eve in P_EVE_GRID:
        raw_ps = [raw[p_eve][1.0][i][0] / 100.0 for i in range(n)]
        per_f  = {f: [raw[p_eve][f][i][0] for i in range(n)] for f in F_SCALES}
        rk, zk = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            raw_q = float(np.mean([raw_ps[i] for i in idx]))
            rk.append(max(0.0, N_SIFTED_FIXED * (1 - 2*h2(raw_q))))
            fm = [float(np.mean([per_f[f][i] for i in idx])) for f in F_SCALES]
            zne_q = zne_estimate_linear(F_SCALES, fm) / 100.0
            zk.append(max(0.0, N_SIFTED_FIXED * (1 - 2*h2(zne_q))))
        rows[p_eve] = dict(
            raw_mean=float(np.mean(rk)),
            raw_lo=float(np.percentile(rk, 2.5)),
            raw_hi=float(np.percentile(rk, 97.5)),
            zne_mean=float(np.mean(zk)),
            zne_lo=float(np.percentile(zk, 2.5)),
            zne_hi=float(np.percentile(zk, 97.5)),
        )
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# NEW v3: Plot helpers
# ═════════════════════════════════════════════════════════════════════════════

def plot_zne_extrapolation(raw40, ic40, linear_extrapolate, out_path="images/E11a_zne_extrapolation_hq.png"):
    """
    Figure 2: ZNE extrapolation plot.
    Points = measured mean QBER at each f.
    Dashed lines = weighted linear fits extrapolated to f=0.
    Dotted lines = true eavesdropper-only QBER (p_eve/4).
    No title (matches paper style).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Color palette matching the uploaded figure (purple → yellow-green)
    colors = ["#440154", "#31688e", "#35b779", "#21918c", "#fde725"]
    # Use a perceptually uniform palette: match the uploaded image order
    # p_eve = 0.0, 0.1, 0.2, 0.3, 0.4
    palette = ["#482878", "#2d708e", "#20a486", "#5dc863", "#fde725"]

    fig, ax = plt.subplots(figsize=(8, 6))

    f_plot = np.linspace(0.0, max(F_SCALES) + 0.05, 200)

    for i, p_eve in enumerate(P_EVE_GRID):
        color = palette[i]
        label = f"p_eve={p_eve}"

        # Measured mean QBER at each f (filled circles)
        fv, qmeans, wts = [], [], []
        for f in F_SCALES:
            if f not in raw40[p_eve]:
                continue
            vals = raw40[p_eve][f]
            qbers = [v[0] for v in vals]
            hws   = [max(1e-3, (v[2]-v[1])/2) for v in vals]
            fv.append(f)
            qmeans.append(float(np.mean(qbers)))
            wts.append(1.0 / float(np.mean(hws))**2)

        ax.scatter(fv, qmeans, color=color, s=80, zorder=5)

        # Weighted linear fit extrapolated to f=0
        a, b = linear_extrapolate(fv, qmeans, wts)
        y_line = [a + b * f for f in f_plot]
        ax.plot(f_plot, y_line, color=color, linestyle="--", linewidth=1.2,
                label=label)

        # True eavesdropper-only QBER (dotted horizontal)
        true_q = p_eve * 25.0
        ax.axhline(y=true_q, color=color, linestyle=":", linewidth=0.9, alpha=0.7)

    ax.set_xlabel("noise-scaling factor f", fontsize=12)
    ax.set_ylabel("measured QBER (%)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, max(F_SCALES) + 0.1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


def plot_key_length_recovery(key_rows, out_path="images/E11f_key_length_recovery.png"):
    """
    Figure 5: Secure key length bar chart.
    Grey bars = Raw QBER key length.
    Blue bars = ZNE-corrected QBER key length.
    Error bars = 95% CI from bootstrap.
    Matches the uploaded figure style.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_eves = P_EVE_GRID
    x = np.arange(len(p_eves))
    width = 0.35

    raw_means  = [key_rows[p]["raw_mean"]  for p in p_eves]
    raw_los    = [key_rows[p]["raw_lo"]    for p in p_eves]
    raw_his    = [key_rows[p]["raw_hi"]    for p in p_eves]
    zne_means  = [key_rows[p]["zne_mean"]  for p in p_eves]
    zne_los    = [key_rows[p]["zne_lo"]    for p in p_eves]
    zne_his    = [key_rows[p]["zne_hi"]    for p in p_eves]

    # Error bar sizes (distance from mean to CI bound)
    raw_err_lo = [m - l for m, l in zip(raw_means, raw_los)]
    raw_err_hi = [h - m for m, h in zip(raw_means, raw_his)]
    zne_err_lo = [m - l for m, l in zip(zne_means, zne_los)]
    zne_err_hi = [h - m for m, h in zip(zne_means, zne_his)]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars_raw = ax.bar(x - width/2, raw_means, width,
                      yerr=[raw_err_lo, raw_err_hi],
                      color="#888888", capsize=4, label="Raw QBER",
                      error_kw={"elinewidth": 1.2, "ecolor": "black"})

    bars_zne = ax.bar(x + width/2, zne_means, width,
                      yerr=[zne_err_lo, zne_err_hi],
                      color="#2196F3", capsize=4, label="ZNE-corrected QBER",
                      error_kw={"elinewidth": 1.2, "ecolor": "black"})

    ax.set_xlabel("pEve", fontsize=12)
    ax.set_ylabel("Secure key length (bits)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in p_eves])
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# C2 — Determine tab:r1 trial count from FER denominators
# ═════════════════════════════════════════════════════════════════════════════

def check_c2_trial_count():
    print("\n" + "=" * 70)
    print("C2 CHECK — tab:r1 trial count (FER denominator diagnostic)")
    print("=" * 70)

    fer_values = {
        "Amplitude damping (unaware)":    0.167,
        "Amplitude damping (aware)":      0.017,
        "Combined T1+T2  (unaware)":      0.083,
        "Combined T1+T2  (aware)":        0.033,
        "Depolarizing     (unaware)":     0.133,
        "Depolarizing     (aware)":       0.050,
        "Phase damping    (unaware)":     0.133,
        "Phase damping    (aware)":       0.000,
    }

    nonzero = {k: v for k, v in fer_values.items() if v > 0}

    print(f"\n  FER values to match (non-zero): {list(nonzero.values())}")
    print(f"\n  {'Denominator':>12}  {'Numerators':>40}  {'All exact?':>12}")
    print("  " + "-" * 68)

    candidates = []
    for denom in [20, 30, 40, 50, 60, 80, 100, 120]:
        nums = {k: round(v * denom) for k, v in nonzero.items()}
        residuals = [abs(nums[k] / denom - v) for k, v in nonzero.items()]
        exact = all(r < 5e-4 for r in residuals)
        num_str = str([nums[k] for k in nonzero])
        mark = "  ✓  MATCH" if exact else ""
        print(f"  {denom:>12}  {num_str:>40}  {str(exact):>12}{mark}")
        if exact:
            candidates.append(denom)

    print()
    if candidates:
        best = candidates[0]
        print(f"  CONCLUSION: tab:r1 FER values are exact fractions of {best}.")
        if best != 40:
            print(f"  *** Caption says '40 paired trials/cell' but the data says {best}. ***")
            print(f"  ACTION: Change tab:r1 caption from '40 paired trials/cell'")
            print(f"          to '{best} paired trials/cell'.")
        else:
            print(f"  Caption ('40 paired trials/cell') is consistent with the data.")
    else:
        print("  No clean denominator found.")

    return candidates[0] if candidates else None


# ═════════════════════════════════════════════════════════════════════════════
# C4 — Verify phase_damping_error Kraus convention
# ═════════════════════════════════════════════════════════════════════════════

def check_c4_phase_damping():
    print("\n" + "=" * 70)
    print("C4 CHECK — phase_damping_error(λ) Kraus convention")
    print("=" * 70)

    try:
        from qiskit_aer.noise import phase_damping_error
        import numpy as np

        lam = 0.1
        err = phase_damping_error(lam)

        kraus_ops = None
        try:
            chan = err.to_quantumchannel()
            kraus_ops = list(chan.data)
        except Exception:
            pass

        if kraus_ops is None:
            try:
                kraus_ops = [np.array(k) for k in err.to_instruction().definition.data
                             if hasattr(k, '__len__')]
            except Exception:
                pass

        if kraus_ops is None:
            from qiskit.quantum_info import Kraus
            k_chan = Kraus(err.to_quantumchannel())
            kraus_ops = k_chan.data

        print(f"\n  Test value: λ = {lam}")
        print(f"  Expected K0[1,1] = sqrt(1 - λ) = {math.sqrt(1 - lam):.6f}")
        print(f"  Expected K1[1,1] = sqrt(λ)     = {math.sqrt(lam):.6f}")
        print()

        found_k0_diag = None
        for i, k in enumerate(kraus_ops):
            k = np.array(k, dtype=complex)
            real_k = k.real
            print(f"  K{i} = {real_k.tolist()}")
            if abs(real_k[0, 0]) > 0.9:
                found_k0_diag = real_k[1, 1]

        print()
        if found_k0_diag is not None:
            expected = math.sqrt(1 - lam)
            match = abs(found_k0_diag - expected) < 1e-6
            print(f"  K0[1,1] from Qiskit : {found_k0_diag:.8f}")
            print(f"  sqrt(1 - λ)         : {expected:.8f}")
            print(f"  Convention match    : {match}")

    except ImportError as e:
        print(f"  Import error: {e}")
    except Exception as e:
        print(f"  Unexpected error: {type(e).__name__}: {e}")
        print("  Paste this error into your message.")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PAPER FIX RERUN v3 — generates corrected numbers + figures")
    print("=" * 70)

    run_zne_sweep, linear_extrapolate, zne_estimate_linear, NoiseModelType = import_project()

    print("\nLoading simulation data …")
    raw20 = get_raw(SEEDS_20, len(SEEDS_SEC), run_zne_sweep, NoiseModelType)
    raw40 = get_raw(SEEDS_40, len(SEEDS_20),  run_zne_sweep, NoiseModelType)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 1
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FIX 1 — Corrected Table 4 (tab:zne-main), 40-seed run")
    print("=" * 70)

    ic40 = intercepts_from_raw(raw40, len(SEEDS_40), linear_extrapolate)

    hdr = (f"{'pEve':>5} | {'True%':>6} | {'Raw f=1':>8} | "
           f"{'ZNE clip':>9} | {'ZNE raw':>9} | "
           f"{'Bias clip':>10} | {'Bias raw':>9}")
    print(hdr)
    print("-" * len(hdr))
    raw_biases_40, zne_biases_40 = [], []
    for p_eve in P_EVE_GRID:
        r = ic40.get(p_eve)
        if not r: continue
        raw_biases_40.append(abs(r["raw_f1"] - r["true_q"]))
        zne_biases_40.append(r["bias_unclipped"])
        print(f"{p_eve:>5.1f} | {r['true_q']:>6.2f} | {r['raw_f1']:>7.2f}% | "
              f"{r['a_clipped']:>8.2f}% | {r['a_raw']:>8.2f}% | "
              f"{r['bias_clipped']:>9.2f}pp | {r['bias_unclipped']:>8.2f}pp")

    mean_raw = float(np.mean(raw_biases_40))
    mean_zne = float(np.mean(zne_biases_40))
    reduction = 100.0 * (1 - mean_zne / mean_raw)
    print(f"\nMean |raw bias|             : {mean_raw:.2f} pp")
    print(f"Mean |ZNE bias| (unclipped) : {mean_zne:.2f} pp")
    print(f"Bias reduction (unclipped)  : {reduction:.1f}%")

    from bb84_zne import linear_extrapolate as _le
    red_mean, red_lo, red_hi = bootstrap_reduction(raw40, SEEDS_40, _le, zne_estimate_linear)
    print(f"Bootstrap 95% CI            : [{red_lo:.1f}%, {red_hi:.1f}%]  (mean {red_mean:.1f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 2
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FIX 2 — Corrected Table 3 (tab:stoptests)")
    print("=" * 70)

    wil = all_three_wilcoxon_tests(raw20, raw40, linear_extrapolate)

    print(f"\n{'Test':>50} | {'n':>5} | {'p-value':>12} | Valid?")
    print("-" * 75)
    print(f"{'OLD (invalid): pooled n×p_eve cells':>50} | "
          f"{wil['invalid_n200']['n']:>5} | {wil['invalid_n200']['p']:>12.2e} | NO")
    print(f"{'Pilot (inconclusive): 20-seed per-seed':>50} | "
          f"{wil['pilot_n20']['n']:>5} | {wil['pilot_n20']['p']:>12.2e} | YES")
    print(f"{'Corrected: 40-seed per-seed (HEADLINE)':>50} | "
          f"{wil['headline_n40']['n']:>5} | {wil['headline_n40']['p']:>12.2e} | YES")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 3
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FIX 3 — No-false-alarm claim: unclipped intercepts")
    print("=" * 70)

    p0 = ic40.get(0.0, {})
    if p0:
        print(f"\n  p_eve=0, depolar={BASE_DEPOLAR}, N={N_QUBITS}, 40 seeds")
        print(f"  Raw QBER at f=1        : {p0['raw_f1']:.2f}%")
        print(f"  ZNE clipped  (max 0,a) : {p0['a_clipped']:.2f}%")
        print(f"  ZNE unclipped (raw a)  : {p0['a_raw']:.2f}%")
        print(f"  Bias (unclipped)       : {p0['bias_unclipped']:.2f} pp")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 4
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"FIX 4 — Corrected Table 5 (tab:keylength), N_sifted={N_SIFTED_FIXED}, 40 seeds")
    print("=" * 70)

    key_rows = bootstrap_key_length(raw40, SEEDS_40, zne_estimate_linear)

    print(f"\n{'pEve':>5} | {'Raw mean [lo, hi]':>26} | {'ZNE mean [lo, hi]':>26} | {'Δ mean':>8} | {'Δ %':>7}")
    print("-" * 85)
    for p_eve in P_EVE_GRID:
        r = key_rows.get(p_eve)
        if not r: continue
        delta = r["zne_mean"] - r["raw_mean"]
        pct = (delta / r["raw_mean"] * 100) if r["raw_mean"] > 0 else float("inf")
        print(f"{p_eve:>5.1f} | {r['raw_mean']:>6.1f} [{r['raw_lo']:>5.1f},{r['raw_hi']:>5.1f}] | "
              f"{r['zne_mean']:>6.1f} [{r['zne_lo']:>5.1f},{r['zne_hi']:>5.1f}] | "
              f"{delta:>+8.1f} | {pct:>6.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 5
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FIX 5 — RQ6 adaptive Eve bias (text says 5.00pp, should be 3.55pp)")
    print("=" * 70)

    P_EVE_BASE = 0.2
    true_contrib = P_EVE_BASE * 25.0
    any_found = False
    for name, exp in [("Static Eve (paper's model)", 0.35),
                      ("LinearScalingEve (f-mimicking)", 3.55),
                      ("BasisBiasedEve (bias=0.85)", 0.36),
                      ("LinearScalingEve (f-mimicking) + RandomF", 0.45)]:
        cache_key = f"adaptive_{name}_{_tag(20, 10)}"
        cached = load_cache(cache_key)
        if cached is not None:
            zne_est = cached.get("zne_linear", float("nan"))
            bias = abs(zne_est - true_contrib)
            print(f"  {name}:")
            print(f"    ZNE estimate = {zne_est:.2f}%,  true = {true_contrib:.2f}%,  bias = {bias:.2f} pp")
            any_found = True

    if not any_found:
        print("  No cached RQ6 results found. Run Section 4 of bb84_phase5_hardening.ipynb.")

    # ─────────────────────────────────────────────────────────────────────────
    # Pure-LaTeX summary
    # ─────────────────────────────────────────────────────────────────────────
    def h2_exact(p):
        if p <= 0 or p >= 1: return 0.0
        return -p*math.log2(p) - (1-p)*math.log2(1-p)
    correct_h2 = 1 - h2_exact(0.65)

    print("\n" + "=" * 70)
    print("PURE-LATEX FIXES (no new simulations needed)")
    print("=" * 70)
    print(f"""
B1. "19 of 20 pool×matrix" → "9 of 10"
B2. Scheme A QBER label swap in Section 4.3 text (+6.0pp↔+5.2pp)
B3. Exact-rate example: 1−h₂(0.65) = {correct_h2:.4f}  (paper says 0.08)
B4. "never once degrading" → "never once degrading in any of the 18,512 paired records"
    "adopt unconditionally" → "adopt without qualification within the validated range"
B5. ZNE scaling map: "maps directly onto" → "is qualitatively analogous to"
B6. RandomF scope: "closes the one identified adversarial vulnerability"
    → "closes it against an adversary with no side-channel access to the noise-scaling schedule"
B7. SECURE label: add caveat that low QBER alone is not a complete security cert
B8. Add sentence explaining 75.9% vs direct-table calculation (~94.9%)
""")

    # ─────────────────────────────────────────────────────────────────────────
    # C2 and C4 checks
    # ─────────────────────────────────────────────────────────────────────────
    c2_denom = check_c2_trial_count()
    check_c4_phase_damping()

    # ─────────────────────────────────────────────────────────────────────────
    # NEW v3: Generate Figure 2 and Figure 5
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PLOT GENERATION — Figure 2 and Figure 5")
    print("=" * 70)

    print("\n  Generating Figure 2 (ZNE extrapolation) …")
    plot_zne_extrapolation(
        raw40, ic40, linear_extrapolate,
        out_path="images/E11a_zne_extrapolation_hq.png"
    )

    print("  Generating Figure 5 (Key length recovery) …")
    plot_key_length_recovery(
        key_rows,
        out_path="images/E11f_key_length_recovery.png"
    )

    print("\n  Both figures saved. Copy them over your existing images/ folder")
    print("  then recompile the LaTeX to update the PDF.")

    # ─────────────────────────────────────────────────────────────────────────
    # Save JSON
    # ─────────────────────────────────────────────────────────────────────────
    out = {
        "generated_by": "rerun_paper_fixes_v3.py",
        "table4_tab_zne_main_40seed": {
            str(p): {k: round(float(v), 4) for k, v in ic40[p].items()}
            for p in P_EVE_GRID if p in ic40
        },
        "headline_bias_stats_40seed": {
            "mean_raw_bias_pp": round(mean_raw, 3),
            "mean_zne_bias_pp_unclipped": round(mean_zne, 3),
            "bias_reduction_pct_unclipped": round(reduction, 2),
            "bootstrap_mean_pct": round(red_mean, 2),
            "bootstrap_ci95_lo_pct": round(red_lo, 2),
            "bootstrap_ci95_hi_pct": round(red_hi, 2),
        },
        "table3_tab_stoptests": {
            "row_invalid_n200": {"n": wil["invalid_n200"]["n"],
                                  "p": f"{wil['invalid_n200']['p']:.2e}"},
            "row_pilot_n20":    {"n": wil["pilot_n20"]["n"],
                                  "p": f"{wil['pilot_n20']['p']:.2e}"},
            "row_headline_n40": {"n": wil["headline_n40"]["n"],
                                  "p": f"{wil['headline_n40']['p']:.2e}"},
        },
        "table5_tab_keylength_40seed": {
            str(p): {k: round(float(v), 1) for k, v in key_rows[p].items()}
            for p in P_EVE_GRID if p in key_rows
        },
        "latex_only_fixes": {
            "rq6_text_bias_correct_value_pp": 3.55,
            "pool_matrix_count_correct": "5 pools x 2 matrices = 10 (not 20)",
            "h2_065_correct_value": round(correct_h2, 4),
            "scheme_A_swap": "swap +6.0pp and +5.2pp in Section 4.3 text to match Table 13",
        },
        "c2_tab_r1_trial_count": {
            "inferred_denominator": c2_denom,
            "caption_says": 40,
            "action": (f"Change caption to '{c2_denom} paired trials/cell'"
                       if c2_denom and c2_denom != 40
                       else "Caption is consistent with data"),
        },
    }

    with open("paper_fix_numbers.json", "w") as fh:
        json.dump(out, fh, indent=2, default=lambda x: round(float(x), 6)
                  if isinstance(x, (float, np.floating)) else x)

    print("\n" + "=" * 70)
    print("✓  All corrected numbers written to:  paper_fix_numbers.json")
    print("✓  Figure 2 written to:  images/E11a_zne_extrapolation_hq.png")
    print("✓  Figure 5 written to:  images/E11f_key_length_recovery.png")
    print("=" * 70)


if __name__ == "__main__":
    main()