"""Independently reconstruct the fixed-code choices from development outputs.

This audit does not call the pipeline training or candidate-selection routines.
It verifies both stored development yields and the deterministic winning row
fraction for every family and block length.
"""
import json
from pathlib import Path

import numpy as np


def fixed_reference_audit(result_dir=None, protocol=None):
    here = Path(__file__).resolve().parent
    out = Path(result_dir) if result_dir is not None else here / "results"
    cfg = protocol if protocol is not None else json.loads((here / "protocol.json").read_text())
    frozen = json.loads((out / "frozen_pipeline_policies.json").read_text())
    development = cfg["pipeline_development"]
    q_index = development["qber"].index(0.06)
    q = development["qber"][q_index]
    entropy = -q * np.log2(q) - (1 - q) * np.log2(1 - q)
    pool = np.asarray(development["rate_pool"], dtype=float)
    rates = sorted({float(pool[np.argmin(abs(pool - margin * entropy))])
                    for margin in development["candidate_margins"]})
    audit = []
    for family in ["gallager", "irregular"]:
        for n in cfg["codes"]["lengths"]:
            yields = {}
            for beta in rates:
                m = 3 * round(beta * n / 3) if family == "gallager" else round(beta * n)
                values = []
                for matrix_seed in cfg["codes"]["development_seeds"]:
                    for ai, _ in enumerate(development["asymmetry"]):
                        for si, sample in enumerate(development["sample"]):
                            stem = f"develop_{family}_n{n}_k{matrix_seed}_q{q_index}_a{ai}_s{si}"
                            with np.load(out / f"{stem}_m{m}_evaluation.npz", allow_pickle=False) as record:
                                assert int(record["m"]) == m and int(record["n"]) == n, stem
                                assert str(record["methods"][0]) == "pooled", stem
                                values.extend((record["correct"][0] * (n - m - 64) / (n + sample)).tolist())
                yields[str(beta)] = float(np.mean(values))
            winner = max(rates, key=lambda beta: (yields[str(beta)], -beta))
            key = f"{family}_{n}"
            saved = frozen["fixed_reference"][key]
            assert yields == saved["development_yields"], (key, "development_yields")
            assert winner == saved["beta"], (key, "beta")
            audit.append(dict(family=family, n=n, beta=winner,
                              development_yields=yields,
                              calibration_qber=q,
                              calibration_asymmetries=development["asymmetry"],
                              status="exact independent fixed-reference selection reproduced"))
    return audit


if __name__ == "__main__":
    print(json.dumps(fixed_reference_audit(), indent=2))
