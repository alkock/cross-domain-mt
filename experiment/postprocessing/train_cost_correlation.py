import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr, pearsonr as _pr

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_BASE = Path(__file__).parent.parent / "output"
FOLDS = [f"fold_{i:02d}" for i in range(12)]
FEATURE_MODELS = ["resnet18", "resnet50", "dinov2", "vgg16"]

POST_HOC_METRICS = ["PSNR", "SSIM", "LPIPS", "FID", "KID_Mean"]


@dataclass
class ExperimentConfig:
    name: str
    output_root: Path


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "heihc": ExperimentConfig(
        name="heihc",
        output_root=OUTPUT_BASE / "heihc_fold",
    ),
    "saropt": ExperimentConfig(
        name="saropt",
        output_root=OUTPUT_BASE / "saropt_fold",
    ),
}


def load_pre_cost(fold_dir: Path, backbone: str, cost_name: str) -> float:
    path = fold_dir / "pre_costs" / backbone / f"{cost_name}_mean.csv"
    df = pd.read_csv(path, index_col=0)
    cols = df.columns.tolist()
    return float(df.loc[cols[0], cols[1]])


def load_test_metrics(fold_dir: Path) -> dict:
    df = pd.read_csv(fold_dir / "metrics.csv")
    row = df[df["Epoch"] == "test"].iloc[0]
    return {m: float(row[m]) for m in POST_HOC_METRICS}


def compute_correlations(cfg: ExperimentConfig) -> pd.DataFrame:
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

    records = []
    for fold in FOLDS:
        fold_dir = cfg.output_root / fold
        entry = {"fold": fold}
        for bb, names in COST_NAMES.items():
            for name in names:
                entry[f"{bb}/{name}"] = load_pre_cost(fold_dir, bb, name)
        entry.update(load_test_metrics(fold_dir))
        records.append(entry)

    df = pd.DataFrame(records).set_index("fold")

    all_cost_cols = [f"{bb}/{name}" for bb in FEATURE_MODELS for name in COST_NAMES[bb]]

    corr_rows = []
    for cost_col in all_cost_cols:
        for metric in POST_HOC_METRICS:
            x = df[cost_col].values
            y = df[metric].values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, p = spearmanr(x, y)
                pearson_r_val, pearson_p = _pr(x, y)
            corr_rows.append(
                {
                    "cost": cost_col,
                    "metric": metric,
                    "pearson_r": round(float(pearson_r_val), 4),
                    "pearson_p": round(float(pearson_p), 4),
                    "spearman_r": round(float(rho), 4),
                    "spearman_p": round(float(p), 4),
                    "n": len(FOLDS),
                }
            )

    corr_df = pd.DataFrame(corr_rows)
    out_path = cfg.output_root / "train_cost_correlation.csv"
    corr_df.to_csv(out_path, index=False)
    print(f"\nCorrelations saved to {out_path}")

    sig = corr_df[corr_df["spearman_p"] < 0.05].sort_values("spearman_p")
    print(
        f"\nSignificant Spearman correlations (p < 0.05): {len(sig)} / {len(corr_df)}"
    )
    if not sig.empty:
        print(
            sig[["cost", "metric", "spearman_r", "spearman_p"]].to_string(index=False)
        )
    else:
        print("None found.")

    return corr_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()),
        required=True,
    )
    args = parser.parse_args()

    cfg = EXPERIMENTS[args.experiment]
    compute_correlations(cfg)
