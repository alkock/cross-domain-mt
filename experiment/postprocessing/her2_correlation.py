"""
Analyses whether higher HER2 score correlates with higher feature-space distance
on the validation split. Each fold has exactly one validation WSI; the WSI ID
maps to a slide ID which maps to a HER2 score.

The output is stored in the "her2_distance_correlation.csv" file in the experiment output directory.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kruskal

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_ROOT = Path(__file__).parent.parent / "output" / "heihc_fold"
FOLDS = [f"fold_{i:02d}" for i in range(12)]
FEATURE_MODELS = ["resnet18", "resnet50", "dinov2", "vgg16"]

# HER2 scores per slide ID  (0, 1, 2, 3 for 0 / 1+ / 2+ / 3+).
HER2_SCORES = {
    1: 0,
    2: 0,
    3: 0,
    5: 3,
    7: 0,
    8: 1,
    9: 3,
    10: 3,
    12: 2,
    13: 3,
    15: 1,
    16: 0,
    18: 1,
    21: 1,
    22: 0,
    25: 2,
    28: 2,
}

COST_NAMES = {
    bb: [
        "centroid",
        "mean",
        "gaussian_mmd",
        f"sinkhorn_{bb}_cosine",
        f"sinkhorn_{bb}_sqeuclidean",
    ]
    for bb in FEATURE_MODELS
}


def extract_val_wsi(fold_dir: Path) -> str:
    """Return the single WSI prefix (e.g. 'wsi021') for a fold's val split."""
    paths = (fold_dir / "val_files.txt").read_text().splitlines()
    wsis = {Path(p).stem.split("_id")[0] for p in paths}
    assert len(wsis) == 1, f"Expected 1 val WSI, got {wsis}"
    return wsis.pop()


def wsi_to_slide_id(wsi: str) -> int:
    """'wsi021' -> 21"""
    return int(wsi.replace("wsi", ""))


def load_val_cost(fold_dir: Path, backbone: str, cost_name: str) -> float:
    path = fold_dir / "val_costs" / backbone / f"{cost_name}_mean.csv"
    df = pd.read_csv(path, index_col=0)
    cols = df.columns.tolist()
    return float(df.loc[cols[0], cols[1]])


def build_table() -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        fold_dir = OUTPUT_ROOT / fold
        wsi = extract_val_wsi(fold_dir)
        slide_id = wsi_to_slide_id(wsi)
        her2 = HER2_SCORES.get(slide_id)

        entry = {
            "fold": fold,
            "wsi": wsi,
            "slide_id": slide_id,
            "her2_score": her2,
        }
        for bb, names in COST_NAMES.items():
            for name in names:
                entry[f"{bb}/{name}"] = load_val_cost(fold_dir, bb, name)

        rows.append(entry)

    return pd.DataFrame(rows).set_index("fold")


def run_correlations(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["her2_score"])
    n_total = len(df)
    n_valid = len(valid)
    if n_valid < n_total:
        print(
            f"Note: {n_total - n_valid} fold(s) dropped due to missing HER2 score "
            f"(wsis: {df[df['her2_score'].isna()]['wsi'].tolist()})\n"
        )

    her2 = valid["her2_score"].values.astype(float)

    all_cost_cols = [f"{bb}/{name}" for bb in FEATURE_MODELS for name in COST_NAMES[bb]]

    rows = []
    for cost_col in all_cost_cols:
        dist = valid[cost_col].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, sp = spearmanr(her2, dist)
            r, pp = pearsonr(her2, dist)

        groups = [dist[her2 == g] for g in np.unique(her2) if (her2 == g).sum() > 0]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kw_stat, kw_p = kruskal(*groups)
        else:
            kw_stat, kw_p = np.nan, np.nan

        rows.append(
            {
                "cost": cost_col,
                "spearman_r": round(float(rho), 4),
                "spearman_p": round(float(sp), 4),
                "pearson_r": round(float(r), 4),
                "pearson_p": round(float(pp), 4),
                "kruskal_stat": (
                    round(float(kw_stat), 4) if not np.isnan(kw_stat) else None
                ),
                "kruskal_p": round(float(kw_p), 4) if not np.isnan(kw_p) else None,
                "n": n_valid,
            }
        )

    return pd.DataFrame(rows)


def print_summary(corr_df: pd.DataFrame) -> None:
    sig = corr_df[corr_df["spearman_p"] < 0.05].sort_values("spearman_p")
    print(f"Significant Spearman correlations (p < 0.05): {len(sig)} / {len(corr_df)}")
    if not sig.empty:
        print(sig[["cost", "spearman_r", "spearman_p"]].to_string(index=False))
    else:
        print("None found.")

    print()
    sig_kw = corr_df[corr_df["kruskal_p"] < 0.05].sort_values("kruskal_p")
    print(f"Significant Kruskal-Wallis (p < 0.05): {len(sig_kw)} / {len(corr_df)}")
    if not sig_kw.empty:
        print(sig_kw[["cost", "kruskal_stat", "kruskal_p"]].to_string(index=False))
    else:
        print("None found.")


def main():
    df = build_table()
    corr_df = run_correlations(df)
    out_path = OUTPUT_ROOT / "her2_distance_correlation.csv"
    corr_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
