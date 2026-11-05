"""Do the explanations computed on this server match those of the thesis machine?

    python code/scripts/remote/check_xai.py

The bundle ships one explained run as a reference, in xai_reference/ (MASTER, DJI, T = 20, L = 5, seed 42,
test year 2020, explained on an RTX 5090), with its checkpoint and the DJI data. This script explains the
same run here with scripts/explain.py (about a minute on a GPU; skipped if already done) and compares
the outputs file by file:
- attribution tensors of IG and Saliency, both targets: correlation and largest difference relative to the
  largest attribution;
- global feature importance: Spearman correlation of the rankings;
- permutation importance (seeded, so it should match) and MASTER's own weights: largest difference.
Differences of floating-point size are expected across GPUs; what matters is that the rankings agree.
The report is printed and written to xai_check.txt (to be sent back with the results).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xaifin.config import RESULTS_ROOT

REPO = Path(__file__).resolve().parents[3]
RUN = Path("MASTER", "dji", "sl20_pl5", "seed42", "y2020")
REFERENCE = REPO / "xai_reference"
EXPLAIN = ["--models", "MASTER", "--universes", "dji", "--horizons", "5", "--seeds", "42", "--years", "2020"]
MIN_CORRELATION, MIN_SPEARMAN = 0.999, 0.99


def main() -> None:
    if not (REFERENCE / "ig_joint" / RUN).is_dir():
        sys.exit(f"no reference in {REFERENCE}: it comes with the bundle (make_bundle.py --xai-reference)")
    if not (RESULTS_ROOT / "xai" / "ig_joint" / RUN / "global_pred.parquet").exists():
        subprocess.run([sys.executable, str(REPO / "code" / "scripts" / "explain.py"), *EXPLAIN], check=True)

    lines, ok = [], True

    def report(name: str, value: str, passed: bool) -> None:
        nonlocal ok
        ok &= passed
        lines.append(f"{'ok  ' if passed else 'DIFF'}  {name}: {value}")

    for method in ("ig_joint", "saliency"):
        for target in ("pred", "topk_margin"):
            with np.load(REFERENCE / method / RUN / f"attributions_{target}.npz") as r, \
                    np.load(RESULTS_ROOT / "xai" / method / RUN / f"attributions_{target}.npz") as h:
                same_rows = (r["attr"].shape == h["attr"].shape and (r["ticker"] == h["ticker"]).all()
                             and (r["date"] == h["date"]).all())
                if not same_rows:
                    report(f"{method} {target} attributions", f"different rows {r['attr'].shape} vs {h['attr'].shape}", False)
                    continue
                a, b = r["attr"].astype(np.float64).ravel(), h["attr"].astype(np.float64).ravel()
                corr = np.corrcoef(a, b)[0, 1]
                report(f"{method} {target} attributions", f"correlation {corr:.6f}, largest difference "
                       f"{np.abs(a - b).max() / np.abs(a).max():.2e} of the largest attribution", corr >= MIN_CORRELATION)
            r = pd.read_parquet(REFERENCE / method / RUN / f"global_{target}.parquet")
            h = pd.read_parquet(RESULTS_ROOT / "xai" / method / RUN / f"global_{target}.parquet")
            m = r[r["level"] == "feature"].merge(h[h["level"] == "feature"], on="name", suffixes=("_r", "_h"))
            rho = spearmanr(m["importance_r"], m["importance_h"]).statistic
            report(f"{method} {target} feature ranking", f"Spearman {rho:.4f} on {len(m)} features", rho >= MIN_SPEARMAN)

    r = pd.read_parquet(REFERENCE / "permutation" / RUN / "global.parquet").set_index("family")
    h = pd.read_parquet(RESULTS_ROOT / "xai" / "permutation" / RUN / "global.parquet").set_index("family")
    for column in ("d_RankIC", "topk_change", "score_change"):
        diff = (r[column] - h[column].reindex(r.index)).abs().max()
        report(f"permutation {column}", f"largest difference {diff:.4f} (values up to {r[column].abs().max():.3f})",
               diff <= 0.05 * r[column].abs().max() + 1e-3)

    with np.load(REFERENCE / "intrinsic" / RUN / "weights.npz") as r, \
            np.load(RESULTS_ROOT / "xai" / "intrinsic" / RUN / "weights.npz") as h:
        for key in ("gate", "temporal"):
            diff = np.abs(r[key].astype(np.float64) - h[key].astype(np.float64)).max()
            report(f"intrinsic {key}", f"largest difference {diff:.2e}", diff <= 1e-3)

    lines.append("RESULT: " + ("the explanations match the reference" if ok else
                               "DIFFERENCES above tolerance: report them to Luca before explaining the grid"))
    text = "\n".join(lines)
    print(text)
    (REPO / "xai_check.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
