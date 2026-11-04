"""Agreement between two importance rankings (THESIS_GUIDE.md Step 7).

Inputs are importance vectors (pd.Series indexed by feature, lag, family, ... name; larger = more
important), e.g. one level of xai.aggregate.global_importance. They are compared on the names they
share: Alpha158 and Alpha360 models share no individual feature, so they are compared at the level
of family, source or horizon only, and then on the families both input sets have.

- spearman, kendall: rank correlations of the whole vectors;
- jaccard@k: |top-k(a) & top-k(b)| / |top-k(a) | top-k(b)|, agreement on the most important names;
- rbo: rank-biased overlap (Webber, Moffat and Zobel 2010, extrapolated RBO_ext, eq. 32), an overlap
  of the two rankings weighted towards the top: with persistence p, depth d weighs p^(d-1), so
  p = 0.9 puts about 86% of the weight on the first 10 names.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def aligned(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The two vectors on their common names, in the same order."""
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def jaccard_topk(a: pd.Series, b: pd.Series, k: int) -> float:
    top_a, top_b = set(a.nlargest(k).index), set(b.nlargest(k).index)
    return len(top_a & top_b) / len(top_a | top_b)


def rbo(a: pd.Series, b: pd.Series, p: float = 0.9) -> float:
    """Extrapolated rank-biased overlap of the two rankings (same names, ties broken by order)."""
    ra, rb = list(a.sort_values(ascending=False).index), list(b.sort_values(ascending=False).index)
    depth = min(len(ra), len(rb))
    seen_a, seen_b, overlap, total = set(), set(), 0, 0.0
    for d in range(1, depth + 1):
        x, y = ra[d - 1], rb[d - 1]
        overlap += (x == y) + (x in seen_b) + (y in seen_a)
        seen_a.add(x)
        seen_b.add(y)
        total += overlap / d * p**d
    return float(overlap / depth * p**depth + (1 - p) / p * total)


def agreement(a: pd.Series, b: pd.Series, k: int = 10, p: float = 0.9) -> dict:
    """Spearman, Kendall, Jaccard of the top k (k capped at the number of names) and RBO, on the common names."""
    a, b = aligned(a, b)
    return {"names": len(a), "spearman": spearmanr(a, b)[0], "kendall": kendalltau(a, b)[0],
            "jaccard": jaccard_topk(a, b, min(k, len(a))), "rbo": rbo(a, b, p)}


def agreement_matrix(vectors: dict[str, pd.Series], metric: str = "spearman", **kwargs) -> pd.DataFrame:
    """Pairwise agreement of several vectors (e.g. one per model), as a symmetric table."""
    names = list(vectors)
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for i, x in enumerate(names):
        out.loc[x, x] = 1.0
        for y in names[i + 1:]:
            out.loc[x, y] = out.loc[y, x] = agreement(vectors[x], vectors[y], **kwargs)[metric]
    return out
