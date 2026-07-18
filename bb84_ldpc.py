
"""
bb84_ldpc.py
============
LDPC syndrome-based information reconciliation for BB84 sifted keys.

THEORETICAL BACKGROUND
-----------------------
After sifting, Alice's key and Bob's key differ on a random subset of
positions: from Bob's point of view, his key is Alice's key passed through
a virtual Binary Symmetric Channel (BSC) with crossover probability equal
to the estimated QBER. This is exactly the Slepian-Wolf source-coding
setup: Bob (with side information Y, his own noisy key) wants to recover
Alice's X, and the minimum information Alice must reveal over the public
(authenticated, but *not* secret) channel is the conditional entropy
H(X|Y) = h2(QBER) bits per sifted bit, where

    h2(p) = -p*log2(p) - (1-p)*log2(1-p)          (binary entropy)

is the Shannon limit for this one-way reconciliation problem.  No real
protocol reaches h2(p) exactly; the ratio between what a protocol actually
reveals and this bound is the *reconciliation efficiency*

    f_EC = (bits revealed per sifted bit) / h2(QBER)      (f_EC >= 1)

LDPC syndrome coding approaches this bound closely (f_EC ~ 1.05-1.2 for
well-designed codes), while remaining a single message, one-way protocol
(unlike interactive protocols such as Cascade, which need several rounds
of comparisons and can reveal more control information overall for a
similar residual error rate; see the discussion in the companion notebook
``experiments/LDPC_Error_Reconciliation.ipynb``).

PROTOCOL IMPLEMENTED HERE
--------------------------
1. Alice and Bob agree in advance (this is public, non-secret information)
   on a single sparse parity-check matrix H0 (the "mother code"), built
   with a Progressive-Edge-Growth-style (PEG-style) construction that
   avoids short cycles in the Tanner graph -- see ``build_peg_ldpc``.
2. For the *current* estimated QBER, the mother code is adapted to a
   target rate R = 1 - f_EC * h2(QBER) via puncturing (QBER low, need a
   higher rate / less redundancy) or shortening (QBER high, need a lower
   rate / more redundancy) of a fixed set of positions -- see
   ``LDPCCode.design_rate``.  No new H is built; the same H0 is reused
   for every QBER value the code was designed to cover.
3. Alice computes the syndrome s = H0 @ x (mod 2), length m0, and "sends"
   it over the public classical channel (simulated here simply by
   returning the array -- in a deployed system this would be one
   authenticated classical message).
4. Bob decodes with the log-domain sum-product (belief propagation)
   algorithm over the Tanner graph, using his noisy bits as the channel
   observation and the syndrome as the parity constraint. Decoding success
   is verified explicitly by recomputing H0 @ x_hat (mod 2) and comparing
   to s -- BP is *not* trusted blindly after max_iterations.

Every syndrome bit sent to Bob is public information that must later be
removed from the final secret key by privacy amplification; this module
tracks and reports that leakage (``leaked_bits``) but does not implement
privacy amplification itself (out of scope, see module docstring in the
notebook).

University of Ruhuna - Dept. of Computer Engineering
MIT Licence - see LICENSE
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "binary_entropy",
    "shannon_target_rate",
    "build_peg_ldpc",
    "LDPCCode",
    "LDPCReconciliationResult",
    "reconcile_sifted_key_with_ldpc",
]


# ──────────────────────────────────────────────────────────────────────
# SHANNON / SLEPIAN-WOLF HELPERS
# ──────────────────────────────────────────────────────────────────────

def binary_entropy(p: float) -> float:
    """Binary entropy h2(p) in bits. h2(0) = h2(1) = 0 by convention."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def shannon_target_rate(p: float, f_ec: float = 1.1) -> float:
    """
    Target reconciliation code rate R = k/n for a BSC with crossover
    probability *p*, given reconciliation efficiency target *f_ec* (>= 1).

    R = 1 - f_ec * h2(p), clipped to [0, 1) since a code rate outside
    that range is not physically meaningful.
    """
    r = 1.0 - f_ec * binary_entropy(p)
    return float(np.clip(r, 0.0, 0.999))


# ──────────────────────────────────────────────────────────────────────
# PEG-STYLE (dv, dc)-REGULAR LDPC CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────

def build_peg_ldpc(n: int, dv: int = 3, dc: int = 6, seed: int = 1) -> np.ndarray:
    """
    Build a (dv, dc)-regular LDPC parity-check matrix using a
    Progressive-Edge-Growth-style greedy construction.

    Unlike a naive ``np.random.randint(0, 2, (m, n))`` matrix (which has
    no controlled Tanner-graph structure and typically contains many
    4-cycles, making belief propagation converge poorly or not at all),
    PEG places each edge one at a time, greedily connecting each variable
    node to the check node that is *farthest* in the current Tanner graph
    (i.e. not yet reachable within a few hops), which locally maximises
    girth. Ties (including "no unreached check available") are broken by
    picking the least-loaded check node, which keeps check degrees as
    close to regular as the edge budget allows.

    Parameters
    ----------
    n  : code length (number of variable nodes). Rounded down to the
         nearest multiple of dc so that m = n*dv/dc is an integer and the
         graph can be made exactly (dv, dc)-regular.
    dv : variable node degree (bits participate in dv parity checks).
    dc : check node degree (each parity check touches dc bits).
    seed : RNG seed, used only to break ties.

    Returns
    -------
    H : (m, n) int8 ndarray, m = n*dv/dc.
    """
    if n < dc:
        raise ValueError(f"n={n} must be >= dc={dc}")
    n = (n // dc) * dc                     # snap to a multiple of dc
    m = (n * dv) // dc

    rng = np.random.default_rng(seed)

    var_edges:   List[List[int]] = [[] for _ in range(n)]   # var -> checks
    check_edges: List[List[int]] = [[] for _ in range(m)]   # check -> vars
    check_degree = np.zeros(m, dtype=int)

    for v in range(n):
        for _ in range(dv):
            existing = set(var_edges[v])

            # BFS outward from v over the graph built so far, alternating
            # var -> check -> var hops, to find which checks are already
            # "reachable" (short-cycle risk) vs. untouched (safe, max girth).
            visited_checks: set = set()
            frontier_vars = {v}
            while frontier_vars and len(visited_checks) < m:
                new_checks = set()
                for vv in frontier_vars:
                    for c in var_edges[vv]:
                        if c not in visited_checks:
                            new_checks.add(c)
                if not new_checks:
                    break
                visited_checks |= new_checks
                next_vars = set()
                for c in new_checks:
                    for vv2 in check_edges[c]:
                        next_vars.add(vv2)
                frontier_vars = next_vars

            candidates = [c for c in range(m)
                          if c not in visited_checks and check_degree[c] < dc]
            if not candidates:
                # Whole graph already reachable from v (dense/late stage):
                # fall back to any check with spare capacity.
                candidates = [c for c in range(m)
                              if c not in existing and check_degree[c] < dc]
            if not candidates:
                # Should not happen for a valid (dv, dc, n) combination,
                # but guard against edge-degeneracy rather than crash.
                candidates = [c for c in range(m) if check_degree[c] < dc]

            min_deg = min(check_degree[c] for c in candidates)
            best = [c for c in candidates if check_degree[c] == min_deg]
            chosen = int(best[int(rng.integers(len(best)))])

            var_edges[v].append(chosen)
            check_edges[chosen].append(v)
            check_degree[chosen] += 1

    H = np.zeros((m, n), dtype=np.int8)
    for v in range(n):
        for c in var_edges[v]:
            H[c, v] = 1
    return H


# ──────────────────────────────────────────────────────────────────────
# LDPC CODE: mother matrix + rate adaptation + BP decoder
# ──────────────────────────────────────────────────────────────────────

class LDPCCode:
    """
    A single (dv, dc)-regular "mother" LDPC code, rate-adapted at
    reconciliation time via puncturing / shortening so it can serve a
    whole range of estimated QBER values without ever rebuilding H.

    Parameters
    ----------
    n0 : mother code length (snapped to a multiple of dc).
    dv, dc : regular Tanner-graph degrees. rate0 = 1 - dv/dc.
    seed : RNG seed for the PEG construction and the fixed position
           ordering used to select punctured / shortened positions.
    max_shorten_frac : maximum fraction of the k0 = n0-m0 "information"
        budget that may be shortened. Bounds how low the effective rate
        can go -- real LDPC-reconciliation systems support a *range* of
        QBER with one mother code, not an unbounded one; beyond this
        cap ``design_rate`` reports the request as infeasible.
    max_puncture_frac : cap on how much of the m0 check budget may be
        dropped/punctured. Raising the rate this way *reduces* the
        code's error-correcting margin (fewer enforced constraints per
        real bit), so, counter-intuitively, too much puncturing can make
        even a very low-QBER block fail to decode. Empirically (see the
        companion notebook) this (dv=3, dc=6) code under flooding-schedule
        BP only holds up reliably to about rate ~0.55-0.60 before that
        margin runs out; the conservative default keeps puncturing well
        inside that region rather than chasing the full Shannon rate.
    qber_design_ceiling : hard cutoff on estimated QBER. Above this, the
        code family is not attempted at all -- this is a deliberate,
        explicit "abort" rather than letting BP run and hope: a real
        system would need a different (lower-rate) mother code or an
        interactive protocol beyond this point.
    """

    def __init__(
        self,
        n0: int = 512,
        dv: int = 3,
        dc: int = 6,
        seed: int = 1,
        max_shorten_frac: float = 0.85,
        max_puncture_frac: float = 0.15,
        qber_design_ceiling: float = 0.15,
    ):
        self.dv, self.dc = dv, dc
        self.H0 = build_peg_ldpc(n0, dv=dv, dc=dc, seed=seed)
        self.m0, self.n0 = self.H0.shape
        self.k0 = self.n0 - self.m0
        self.rate0 = self.k0 / self.n0
        self.seed = seed
        self.max_shorten_frac   = max_shorten_frac
        self.max_puncture_frac  = max_puncture_frac
        self.qber_design_ceiling = qber_design_ceiling

        # Fixed (public) orderings used to pick which *variable* positions
        # get shortened, and which *check* rows get punctured, for a given
        # (s, p). Fixed once per code instance so the same mother code is
        # reused across every QBER value -- only s (or p) changes.
        self._position_order = np.random.default_rng(seed + 1).permutation(self.n0)
        self._check_order     = np.random.default_rng(seed + 2).permutation(self.m0)

        self._build_bp_indices()

    # ── Tanner-graph edge bookkeeping for vectorised BP ─────────────

    def _build_bp_indices(self) -> None:
        n0, m0, dv, dc = self.n0, self.m0, self.dv, self.dc
        # var-major edge order: edges of var 0 (dv of them), then var 1, ...
        edge_chk = np.zeros(n0 * dv, dtype=int)
        # recover, for each var, its dv connected checks in a fixed order
        for v in range(n0):
            checks_v = np.nonzero(self.H0[:, v])[0]
            assert len(checks_v) == dv, "H0 is not exactly dv-regular"
            edge_chk[v * dv:(v + 1) * dv] = checks_v

        order = np.argsort(edge_chk, kind="stable")   # var-major -> check-major
        # sanity: every check must end up with exactly dc edges
        chk_sorted = edge_chk[order]
        counts = np.bincount(chk_sorted, minlength=m0)
        assert np.all(counts == dc), "H0 is not exactly dc-regular"

        self._order = order                                  # (n0*dv,)
        self._inv_order = np.argsort(order)                   # (n0*dv,)

    # ── Rate design (puncturing / shortening) ───────────────────────

    def design_rate(self, estimated_qber: float, f_ec: float = 1.5) -> dict:
        """
        Choose (s, p) so the code's effective rate approximates the
        Shannon/Slepian-Wolf target rate for the given estimated QBER,
        reusing the *same* mother matrix H0 in both directions:

        - s = number of **variable** positions shortened (fixed to a
          known value 0, removed from the real secret-key budget). This
          only ever *lowers* the rate -- used when the channel is worse
          than the mother code's native rate0 (high QBER).
        - p = number of **check** rows punctured, i.e. simply not sent /
          not enforced (Alice does not reveal that syndrome bit at all,
          and Bob's decoder drops that parity constraint entirely). Rate
          over the (unchanged) n0 real bits then INCREASES: fewer parity
          bits are spent per real bit. Used when the channel is better
          than rate0 (low QBER).

        Exactly one of (s, p) is nonzero. Returns a dict with keys:
        feasible, s, p, n_real (real variable positions used, n0 - s),
        m_eff (syndrome bits actually sent/enforced, m0 - p),
        target_rate, achieved_rate, reason (set when infeasible).
        """
        p_est = float(np.clip(estimated_qber, 1e-4, 0.5 - 1e-4))

        if estimated_qber > self.qber_design_ceiling:
            return dict(feasible=False, s=0, p=0, n_real=0, m_eff=0,
                        target_rate=None, achieved_rate=None,
                        reason=(f"estimated QBER {estimated_qber:.3f} exceeds this "
                                f"code family's design ceiling "
                                f"{self.qber_design_ceiling:.3f}; a lower-rate "
                                f"mother code (or an interactive protocol) is "
                                f"required beyond this point."))

        target_rate = shannon_target_rate(p_est, f_ec=f_ec)
        n0, m0 = self.n0, self.m0
        max_s = int(self.max_shorten_frac * (self.k0 - 1))
        max_p = int(self.max_puncture_frac * m0)

        if target_rate <= self.rate0:
            # Need MORE redundancy per real bit -> shorten variable positions.
            # R_eff(s) = (n0 - s - m0) / (n0 - s) = target_rate
            #   => s = (n0 - m0 - target_rate*n0) / (1 - target_rate)
            denom = max(1e-9, 1.0 - target_rate)
            s = int(math.ceil((n0 - m0 - target_rate * n0) / denom))
            s = int(np.clip(s, 0, max_s))
            p = 0
            n_real, m_eff = n0 - s, m0
            if n_real <= m_eff:
                return dict(feasible=False, s=s, p=0, n_real=n_real, m_eff=m_eff,
                            target_rate=target_rate, achieved_rate=None,
                            reason="required shortening exceeds max_shorten_frac cap")
            achieved_rate = (n_real - m_eff) / n_real
        else:
            # Need LESS redundancy per real bit -> puncture (drop) check rows.
            # R_eff(p) = (n0 - (m0 - p)) / n0 = target_rate
            #   => p = m0 - n0*(1 - target_rate)
            p = int(math.floor(m0 - n0 * (1.0 - target_rate)))
            p = int(np.clip(p, 0, max_p))
            s = 0
            n_real, m_eff = n0, m0 - p
            achieved_rate = (n_real - m_eff) / n_real

        return dict(feasible=True, s=s, p=p, n_real=n_real, m_eff=m_eff,
                     target_rate=target_rate, achieved_rate=achieved_rate,
                     reason=None)

    # ── Positions / check-mask for a given (s, p), with optional extra
    #    shortening to pad a short final block ───────────────────────

    def _active_var_positions(self, s: int, extra_shorten: int = 0):
        """Return (active, shortened) variable-position index arrays."""
        order = self._position_order
        s_eff = s + extra_shorten
        shortened = order[self.n0 - s_eff:] if s_eff > 0 else order[self.n0:self.n0]
        active = order[:self.n0 - s_eff]
        return active, shortened

    def _active_check_mask(self, p: int) -> np.ndarray:
        """Boolean (m0,) mask, True for the m0-p check rows that are
        actually revealed (syndrome sent) and enforced during decoding."""
        mask = np.ones(self.m0, dtype=bool)
        if p > 0:
            mask[self._check_order[:p]] = False
        return mask

    # ── Belief propagation (log-domain sum-product) ─────────────────

    def bp_decode(self, channel_llr: np.ndarray, syndrome: np.ndarray,
                  active_check_mask: Optional[np.ndarray] = None,
                  max_iter: int = 100, llr_clip: float = 25.0):
        """
        Sum-product (belief propagation) syndrome decoding.

        Parameters
        ----------
        channel_llr : (n0,) array. L = log(P(x=0)/P(x=1)) per position;
                      positive means bit 0 is more likely.
        syndrome    : (m0,) 0/1 array, the *target* syndrome H0 @ x.
        active_check_mask : (m0,) bool array. False entries mark punctured
                      check rows -- their syndrome bit was never revealed,
                      so they contribute no message and are excluded from
                      the convergence test (True = all m0 checks active).
        max_iter    : maximum flooding-schedule iterations.

        Returns
        -------
        x_hat : (n0,) int array, hard-decision estimate.
        converged : bool, True iff H0[active] @ x_hat (mod 2) == syndrome[active]
                    was reached before max_iter (explicitly checked, never
                    assumed from running out of iterations).
        n_iter : number of iterations actually run.
        """
        m0, dc, dv = self.m0, self.dc, self.dv
        order, inv_order = self._order, self._inv_order
        if active_check_mask is None:
            active_check_mask = np.ones(m0, dtype=bool)

        sign_fix = np.where(syndrome % 2 == 1, -1.0, 1.0)   # (-1)^{s_c}, shape (m0,)
        eps = 1e-8

        Mc2v = np.zeros((self.n0, dv))     # check->var messages, var-major (n0, dv)
        x_hat = (channel_llr < 0).astype(int)

        for it in range(1, max_iter + 1):
            total = channel_llr + Mc2v.sum(axis=1)
            M_v2c = total[:, None] - Mc2v                       # (n0, dv)
            M_v2c = np.clip(M_v2c, -llr_clip, llr_clip)

            Mv2c_ck = M_v2c.flatten()[order].reshape(m0, dc)     # check-major

            t = np.tanh(Mv2c_ck / 2.0)
            t = np.clip(t, -1.0 + eps, 1.0 - eps)
            sign = np.sign(t)
            sign[sign == 0] = 1.0
            log_abs = np.log(np.abs(t))

            sum_log_abs = log_abs.sum(axis=1, keepdims=True)
            prod_sign   = np.prod(sign, axis=1, keepdims=True)

            excl_log_abs = sum_log_abs - log_abs                 # (m0, dc)
            excl_sign    = prod_sign * sign                      # (m0, dc)

            tanh_excl = excl_sign * np.exp(excl_log_abs)
            tanh_excl *= sign_fix[:, None]
            tanh_excl = np.clip(tanh_excl, -1.0 + eps, 1.0 - eps)

            Mc2v_ck = 2.0 * np.arctanh(tanh_excl)
            Mc2v_ck = np.clip(Mc2v_ck, -llr_clip, llr_clip)
            Mc2v_ck[~active_check_mask, :] = 0.0    # punctured checks: no message

            Mc2v = Mc2v_ck.flatten()[inv_order].reshape(self.n0, dv)

            marginal = channel_llr + Mc2v.sum(axis=1)
            x_hat = (marginal < 0).astype(int)

            syn = (self.H0 @ x_hat) % 2
            if np.array_equal(syn[active_check_mask], syndrome[active_check_mask]):
                return x_hat, True, it

        return x_hat, False, max_iter


# ──────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LDPCReconciliationResult:
    """Output of ``reconcile_sifted_key_with_ldpc``."""

    success: bool
    """True iff every block converged to a syndrome-consistent codeword."""

    reconciled_key: List[int]
    """Bob's key after reconciliation (fallback to Bob's original bits per
    block where that block failed to converge)."""

    code_rate: float
    """Achieved rate k_real/n_real, averaged (bit-weighted) over blocks."""

    syndrome_length: int
    """Total syndrome bits sent across all blocks (m0 * n_blocks)."""

    num_bp_iterations: float
    """Mean BP iterations to convergence/termination across blocks."""

    converged: bool
    """True iff every block's BP decoder converged before max_iter."""

    leaked_bits: int
    """Total public bits revealed via the syndrome (== syndrome_length
    for this one-way, syndrome-only protocol) -- strip these via privacy
    amplification before using the key."""

    reconciliation_efficiency: float
    """Achieved f_EC = (1 - code_rate) / h2(estimated_qber)."""

    residual_errors: int
    """Oracle-only diagnostic (requires alice_key): bit disagreements
    between reconciled_key and alice_key after reconciliation."""

    # ── extra diagnostics (not required by the spec, but useful) ────
    n_blocks: int = 0
    block_size: int = 0
    estimated_qber: float = 0.0
    feasible: bool = True
    reason: Optional[str] = None
    per_block_converged: List[bool] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# INTEGRATION FUNCTION
# ──────────────────────────────────────────────────────────────────────

def reconcile_sifted_key_with_ldpc(
    bob_key: List[int],
    alice_key: List[int],
    estimated_qber: float,
    f_ec: float = 1.5,
    code: Optional[LDPCCode] = None,
    n0: int = 512,
    dv: int = 3,
    dc: int = 6,
    seed: int = 1,
    max_iter: int = 100,
) -> Tuple[List[int], LDPCReconciliationResult]:
    """
    Reconcile Bob's noisy sifted/final key against Alice's, using LDPC
    syndrome decoding rate-adapted to ``estimated_qber``.

    This is the integration entry point analogous to
    ``correct_sifted_key_with_grand`` in ``bb84_grand.py``: pass in
    ``result.bob_final_key``, ``result.alice_final_key`` and
    ``result.qber_result.qber`` from a ``bb84_runner.run_simulation()``
    call.

    Parameters
    ----------
    bob_key, alice_key : equal-length 0/1 lists (Alice's is used both to
        compute the "public" syndrome, as she legitimately would, and
        -- because this is a simulation -- to verify the result and
        report residual errors).
    estimated_qber : QBER estimate from the public sample
        (``result.qber_result.qber``).
    f_ec : target reconciliation efficiency (>= 1) used to set the design
        rate R = 1 - f_ec*h2(p). Large, near-capacity, degree-optimised
        irregular LDPC codes in the literature reach f_EC ~ 1.05-1.2; the
        modest regular (dv, dc) code built here needs a larger design
        margin to decode reliably at finite block length under simple
        flooding BP (validated empirically in the companion notebook),
        hence the higher default. The *achieved* f_EC is reported
        separately in the result and will typically sit above 1.2 for
        that reason -- see the notebook's discussion section.
    code : reuse an existing LDPCCode (mother matrix) instead of building
        a new one -- saves the PEG-construction cost across repeated
        calls. If None, one is built with the given (n0, dv, dc, seed).

    Returns
    -------
    reconciled_key : Bob's key after correction (or his original bits,
        block-wise, where reconciliation failed).
    result : LDPCReconciliationResult with full metadata.
    """
    assert len(bob_key) == len(alice_key), (
        f"Key length mismatch: Bob={len(bob_key)}, Alice={len(alice_key)}"
    )

    if code is None:
        code = LDPCCode(n0=n0, dv=dv, dc=dc, seed=seed)

    L = len(bob_key)
    if L == 0:
        return [], LDPCReconciliationResult(
            success=True, reconciled_key=[], code_rate=code.rate0,
            syndrome_length=0, num_bp_iterations=0.0, converged=True,
            leaked_bits=0, reconciliation_efficiency=0.0, residual_errors=0,
            n_blocks=0, block_size=0, estimated_qber=estimated_qber,
        )

    design = code.design_rate(estimated_qber, f_ec=f_ec)
    if not design["feasible"]:
        return list(bob_key), LDPCReconciliationResult(
            success=False, reconciled_key=list(bob_key),
            code_rate=code.rate0, syndrome_length=0, num_bp_iterations=0.0,
            converged=False, leaked_bits=0, reconciliation_efficiency=0.0,
            residual_errors=sum(a != b for a, b in zip(alice_key, bob_key)),
            n_blocks=0, block_size=0, estimated_qber=estimated_qber,
            feasible=False, reason=design["reason"],
        )

    s, p, n_real, m_eff = design["s"], design["p"], design["n_real"], design["m_eff"]
    n_blocks = math.ceil(L / n_real)
    active_check_mask = code._active_check_mask(p)

    reconciled: List[int] = []
    total_iters = 0
    all_converged = True
    total_leaked = 0
    total_k_real = 0
    total_n_real = 0
    per_block_converged: List[bool] = []

    for b in range(n_blocks):
        a_blk = list(alice_key[b * n_real:(b + 1) * n_real])
        y_blk = list(bob_key[b * n_real:(b + 1) * n_real])
        extra_shorten = n_real - len(a_blk)      # pad a short final block

        active, shortened = code._active_var_positions(s, extra_shorten)
        n0 = code.n0

        # ── build Alice's full n0-length codeword x ─────────────────
        x = np.zeros(n0, dtype=int)
        x[active] = a_blk
        # shortened positions already 0

        syndrome = (code.H0 @ x) % 2       # full m0-length syndrome; only the
                                            # m_eff active rows are actually
                                            # revealed/enforced below.

        # ── build Bob's channel LLR vector ──────────────────────────
        p_bsc = float(np.clip(estimated_qber, 1e-4, 0.5 - 1e-4))
        base_llr = math.log((1.0 - p_bsc) / p_bsc)

        channel_llr = np.zeros(n0)
        y_active = np.array(y_blk, dtype=int)
        channel_llr[active] = (1 - 2 * y_active) * base_llr
        if len(shortened) > 0:
            channel_llr[shortened] = 25.0     # certain 0 (known/shared padding)

        x_hat, converged, n_iter = code.bp_decode(
            channel_llr, syndrome, active_check_mask=active_check_mask, max_iter=max_iter,
        )

        block_active_len = len(a_blk)          # real bits actually from the key
        if converged:
            reconciled.extend(x_hat[active][:block_active_len].tolist())
        else:
            all_converged = False
            reconciled.extend(y_blk)           # fall back to Bob's uncorrected bits

        per_block_converged.append(bool(converged))
        total_iters += n_iter
        total_leaked += m_eff
        total_k_real += (n_real - m_eff)
        total_n_real += n_real

    residual_errors = sum(a != b for a, b in zip(alice_key, reconciled))
    achieved_rate = total_k_real / total_n_real if total_n_real else 0.0
    h2 = binary_entropy(float(np.clip(estimated_qber, 1e-4, 0.5 - 1e-4)))
    f_ec_achieved = (1.0 - achieved_rate) / h2 if h2 > 0 else float("nan")

    result = LDPCReconciliationResult(
        success=all_converged,
        reconciled_key=reconciled,
        code_rate=achieved_rate,
        syndrome_length=total_leaked,
        num_bp_iterations=total_iters / n_blocks,
        converged=all_converged,
        leaked_bits=total_leaked,
        reconciliation_efficiency=f_ec_achieved,
        residual_errors=residual_errors,
        n_blocks=n_blocks,
        block_size=n_real,
        estimated_qber=estimated_qber,
        feasible=True,
        reason=None,
        per_block_converged=per_block_converged,
    )
    return reconciled, result


# ──────────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────────

def _bsc_corrupt(bits: List[int], p: float, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    flips = rng.random(len(bits)) < p
    arr = np.array(bits, dtype=int)
    arr[flips] ^= 1
    return arr.tolist()


if __name__ == "__main__":
    print("=" * 70)
    print("bb84_ldpc.py -- self-test")
    print("=" * 70)

    print(f"\nBuilding mother LDPC code (n0=512, dv=3, dc=6, PEG construction)...")
    code = LDPCCode(n0=512, dv=3, dc=6, seed=1)
    print(f"  m0={code.m0}  n0={code.n0}  k0={code.k0}  rate0={code.rate0:.3f}")

    # sanity: exactly regular
    col_deg = code.H0.sum(axis=0)
    row_deg = code.H0.sum(axis=1)
    assert np.all(col_deg == code.dv), "variable degree not regular"
    assert np.all(row_deg == code.dc), "check degree not regular"
    print(f"  Tanner graph confirmed exactly ({code.dv},{code.dc})-regular.")

    # Below the design ceiling, a single LDPC block (one BP decode call)
    # at each QBER point must converge to the *correct* codeword at least
    # ~90% of the time over independent BSC draws -- this is exactly the
    # frame-error-rate (FER) metric explored in depth in the companion
    # notebook (Section 2). A handful of failures right at the margin is
    # expected finite-length behaviour, not a bug; what would be a bug is
    # *silent* corruption (success=True but wrong key) or a success rate
    # that never gets close to 90% inside the design range.
    # Above the ceiling, the request must be refused outright (feasible
    # = False) rather than let BP run and silently produce a wrong key.
    below_ceiling = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    above_ceiling = [0.20, 0.25, 0.40]

    n_trials = 30
    key_len = 128     # <= every achievable n_real in this range -> 1 block/trial

    all_ok = True

    print(f"\n--- below design ceiling ({code.qber_design_ceiling:.2f}): "
          f"single-block FER should be low ---")
    for qber in below_ceiling:
        successes = 0
        for t in range(n_trials):
            rng = np.random.default_rng(1000 + t)
            alice_key = rng.integers(0, 2, key_len).tolist()
            bob_key = _bsc_corrupt(alice_key, qber, seed=2000 + int(qber * 1000) + t)

            reconciled, result = reconcile_sifted_key_with_ldpc(
                bob_key, alice_key, estimated_qber=qber, code=code,
            )
            assert result.n_blocks == 1
            if result.success:
                successes += 1
                assert reconciled == alice_key, (
                    "BP reported success but reconciled key does not match "
                    "Alice's key -- silent corruption bug!"
                )
                assert result.residual_errors == 0

        rate = successes / n_trials
        ok = rate >= 0.9
        all_ok &= ok
        print(f"[{'OK  ' if ok else 'FAIL'}] QBER={qber:.3f}  "
              f"success={successes}/{n_trials}  achieved_rate={result.code_rate:.3f}  "
              f"f_EC={result.reconciliation_efficiency:.2f}  leaked_bits={result.leaked_bits}  "
              f"avg_iters={result.num_bp_iterations:.1f}")

    print(f"\n--- above design ceiling: expect a clean, explicit abort ---")
    for qber in above_ceiling:
        rng = np.random.default_rng(42)
        alice_key = rng.integers(0, 2, key_len).tolist()
        bob_key = _bsc_corrupt(alice_key, qber, seed=int(qber * 1000))
        n_errors_in = sum(a != b for a, b in zip(alice_key, bob_key))

        reconciled, result = reconcile_sifted_key_with_ldpc(
            bob_key, alice_key, estimated_qber=qber, code=code,
        )
        ok = (not result.success) and (not result.feasible)
        all_ok &= ok
        print(f"[{'OK  ' if ok else 'FAIL'}] QBER={qber:.2f}  errors_in={n_errors_in}  "
              f"success={result.success}  feasible={result.feasible}  reason={result.reason}")
        # Must NOT silently claim success with a wrong key.
        assert reconciled != alice_key or n_errors_in == 0

    print("\n" + "=" * 70)
    if all_ok:
        print("ALL SELF-TESTS PASSED")
    else:
        print("SELF-TEST FAILURES DETECTED")
    print("=" * 70)

    if not all_ok:
        raise SystemExit(1)
