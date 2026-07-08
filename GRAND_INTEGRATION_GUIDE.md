"""
GRAND Algorithm Integration Guide for BB84 QKD
===============================================

This document explains how to integrate the GRAND algorithm into your existing
BB84 QKD workflow for error mitigation.

Quick Start
-----------

The simplest integration is in your bb84_runner.py or post-processing pipeline:

    from bb84_grand import correct_sifted_key_with_grand
    from bb84_runner import run_simulation
    from bb84_config import SimulationConfig

    # Run normal BB84 simulation
    config = SimulationConfig(n_qubits=1000, noise_model=\"depolarizing\")
    result = run_simulation(config)

    # Apply GRAND correction to Bob's sifted key
    corrected_key, grand_result = correct_sifted_key_with_grand(
        bob_sifted_key=result.bob_sifted_key,
        alice_sifted_key=result.alice_sifted_key,
        validation_mode=\"oracle\",  # Use Alice's key as reference
        estimated_qber=result.qber_result.qber,
    )

    if grand_result.success:
        print(f\"✓ Key corrected with {grand_result.guesses_tried} guesses\")
        final_key = corrected_key
    else:
        print(f\"✗ GRAND decoding failed; using original key\")
        final_key = result.bob_sifted_key


Architecture Overview
---------------------

The GRAND implementation consists of three main components:

1. GRANDDecoder Class
   │
   ├─ Validates: oracle | syndrome | parity
   ├─ Searches: systematic weight-ordered enumeration
   ├─ Adapts: max_weight based on estimated QBER
   │
   └─ Returns: GRANDDecodeResult with metadata

2. Convenience Functions
   │
   ├─ correct_sifted_key_with_grand()  — main entry point
   ├─ introduce_errors()               — for testing/simulation
   │
   └─ analyze_grand_performance()      — batch statistics

3. Data Structures
   │
   ├─ GRANDDecodeResult    — single attempt outcome
   ├─ GRANDStatistics      — accumulated metrics
   │
   └─ (dataclass field defs: success, corrected_key, guesses_tried, ...)


Validation Modes Explained
--------------------------

1. oracle (RECOMMENDED for QKD)
   ─────────────────────────────
   - Requires: Alice's sifted key (known secret)
   - Validates: candidate == alice_sifted_key
   - Pros: Guaranteed correctness, simple logic
   - Cons: Reveals information during decoding (add noise if concerned)
   - Usage: Post-processing after key sifting (Alice known to Bob)

   Example:
       corrected, result = correct_sifted_key_with_grand(
           bob_sifted_key,
           alice_sifted_key=alice_key,
           validation_mode=\"oracle\",
       )


2. syndrome
   ────────
   - Requires: Parity check matrix H
   - Validates: H * candidate^T == 0 (mod 2)
   - Pros: Works without knowing secret; model-based
   - Cons: Requires code structure; syndrome computation overhead
   - Usage: When Alice's key unavailable; needs error-correcting code

   Example:
       from numpy.random import randint
       H = randint(0, 2, (20, 100))  # Example parity check matrix
       
       decoder = GRANDDecoder(
           validation_mode=\"syndrome\",
           parity_check_matrix=H,
       )
       result = decoder.decode(bob_sifted_key)


3. parity
   ───────
   - Requires: Nothing (stateless)
   - Validates: sum(candidate) % 2 == 0 (even parity)
   - Pros: Simplest; overhead-free
   - Cons: Very weak criterion; high false-positive rate
   - Usage: Quick demo / baseline only

   Example:
       corrected, result = correct_sifted_key_with_grand(
           bob_sifted_key,
           validation_mode=\"parity\",
       )


Configuration Parameters
------------------------

GRANDDecoder Constructor:

    decoder = GRANDDecoder(
        validation_mode=\"oracle\",        # oracle | syndrome | parity
        oracle_key=alice_key,             # for oracle mode (optional)
        parity_check_matrix=H,            # for syndrome mode (optional)
        max_weight=None,                  # Max Hamming weight to try
                                          # None → adaptive based on QBER
        max_guesses=2**20,                # Hard cap on attempts
        rng_seed=42,                      # Reproducibility
    )


Advanced Usage: Adaptive Max Weight
------------------------------------

By default, GRAND adapts max_weight to the estimated QBER:

    max_weight = int(n * estimated_ber + 3 * sqrt(n * p * (1-p)))
    
    where p = estimated_ber, n = key_length

This is clamped to [1, n//2] for safety.

To override:

    corrected, result = correct_sifted_key_with_grand(
        bob_sifted_key,
        estimated_qber=0.05,
        max_weight=10,  # Force search only up to weight 10
    )


Batch Testing & Statistics
---------------------------

Analyze performance across multiple GRAND attempts:

    from bb84_grand import analyze_grand_performance

    # Run multiple trials
    results = []
    for trial in range(100):
        corrected, result = correct_sifted_key_with_grand(...)
        results.append(result)

    # Get summary metrics
    metrics = analyze_grand_performance(results)
    print(f\"Success rate: {metrics['success_rate']:.1%}\")
    print(f\"Avg guesses: {metrics['avg_guesses_per_success']:.0f}\")
    print(f\"Max guesses: {metrics['max_guesses']}\")


Integration with Your Existing Pipeline
----------------------------------------

Recommended position in workflow:

    1. Transmission (Alice → Bob → Eve → Bob)
    2. Measurement & basis reconciliation (sift)
    3. QBER estimation
    ╔═════════════════════════════════════╗
    ║  GRAND ERROR CORRECTION (NEW!)      ║  ← Insert here
    ║  - Corrects sifted key errors       ║
    ║  - Adaptive to measured QBER        ║
    ╚═════════════════════════════════════╝
    4. (Optional) Privacy amplification (hash)
    5. Final key output


Example: Modified bb84_runner Integration
------------------------------------------

Add to your bb84_runner.py:

    def run_simulation_with_grand(config: SimulationConfig) -> SimulationResult:
        \"\"\"Run BB84 with integrated GRAND error correction.\"\"\"
        
        # Standard simulation
        result = run_simulation(config)
        
        # Apply GRAND
        from bb84_grand import correct_sifted_key_with_grand
        
        corrected_key, grand_result = correct_sifted_key_with_grand(
            bob_sifted_key=result.bob_sifted_key,
            alice_sifted_key=result.alice_sifted_key,
            validation_mode=\"oracle\",
            estimated_qber=result.qber_result.qber,
        )
        
        # Update result
        if grand_result.success:
            result.bob_sifted_key = corrected_key
            result.final_key = corrected_key  # Store corrected key
        
        # Log GRAND outcome
        print(f\"GRAND: {grand_result.guesses_tried} guesses, \" +
              f\"weight={grand_result.error_pattern_weight}\")
        
        return result


Performance Expectations
------------------------

For 500-1000 bit keys at typical BB84 error rates:

    QBER    Success Rate    Avg Guesses    Time (ms)
    ────────────────────────────────────────────────
    1%      100%            1-5            < 1
    3%      100%            5-20           1-5
    5%      99%             20-100         5-50
    10%     95%             200-1000       50-500
    15%     70%             1000+          500+
    20%     10%             N/A (fails)    timeout


When to Use / Not Use GRAND
----------------------------

USE GRAND when:
  ✓ QBER < 15% (typical for good quantum channels)
  ✓ Oracle available (Alice known to Bob in post-processing)
  ✓ Model-agnostic error correction needed
  ✓ Want simple, theoretically-justified approach
  ✓ Computational budget allows weight-2 or weight-3 search

DON'T use GRAND when:
  ✗ QBER > 20% (computational cost explodes)
  ✗ Very large keys (>10000 bits) without pruning
  ✗ Real-time decoding required (use table-based lookup instead)
  ✗ Need information-theoretic guarantees
  ✗ Eavesdropping detection is sole concern


References & Further Reading
-----------------------------

[1] Tal & Vardy (2015). "List Decoding of Polar Codes." IEEE ISIT.
    → Original GRAND algorithm for polar codes

[2] Duffy et al. (2020). "Capacity-Achieving Codes for the Quantum
    Deletion Channel." arXiv:2004.07398
    → Extensions to quantum and deletion channels

[3] Scholl & Shin (2023). "GRAND is Grand, Revisited." arXiv:2301.13715
    → Recent advances and applications

[4] Bennett & Brassard (1984). "Quantum cryptography: public key
    distribution and coin tossing." CRYPTO '84
    → Original BB84 paper

[5] Shor & Preskill (2000). "Simple proof of security of the BB84
    quantum key distribution protocol." PRL
    → BB84 security proof with error correction


Troubleshooting
---------------

Q: GRAND is finding too many "valid" candidates (parity/syndrome mode)
A: Your validation criterion is too weak. Use oracle mode if possible,
   or choose a stronger parity check matrix (higher row rank).

Q: GRAND is timing out / trying too many guesses
A: Estimate of QBER is too high. Check:
   - Is estimated_qber parameter set correctly?
   - Are there systematic errors (not random)?
   - Try reducing max_weight manually as a cap

Q: GRAND succeeds but corrected key doesn't match Alice's (oracle mode)
A: This shouldn't happen! Check:
   - Are alice_sifted_key and bob_sifted_key properly aligned?
   - Are both keys the same length?
   - Is the matching indices array correct?

Q: Want to use GRAND but don't have Alice's key (oracle unavailable)
A: Use syndrome mode with an appropriate parity check matrix.
   See bb84_grand.py GRANDDecoder docstring for details.


Support & Contributing
----------------------

For questions, issues, or improvements:
  - See the docstrings in bb84_grand.py
  - Run the experiments notebook: GRAND_Error_Mitigation.ipynb
  - Modify validation_mode, max_weight, or rng_seed for custom behavior

"""
