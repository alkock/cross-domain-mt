import sys
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.cost_modules.centroid_distance_estimator as cde
import src.cost_modules.gaussian_mmd_estimator as gmmde
import src.cost_modules.mean_cost_estimator as mce
import src.cost_modules.sinkhorn_distance_estimator as sde
from src.feature_extractor import FeatureExtractor, build_bootstrap_draws


OUT_BASE = Path(__file__).parent.parent / "output"
FOLDS = [f"fold_{i:02d}" for i in range(12)]
BACKBONES = ["resnet18", "resnet50", "dinov2", "vgg16"]
METRICS = ["PSNR", "SSIM", "LPIPS", "FID", "KID_Mean"]

N_REPEATS = 50
N_BOOTSTRAP = 389

experiments = [
    (
        "heihc",
        OUT_BASE / "heihc_fold",
        "HE",
        "IHC",
        lambda p: Path(str(p).replace("/HE/", "/IHC/")),
    ),
    (
        "saropt",
        OUT_BASE / "saropt_fold",
        "SAR",
        "OPT",
        lambda p: Path(str(p).replace("/input/", "/target/")),
    ),
]

extractors = {bb: FeatureExtractor(model_name=bb) for bb in BACKBONES}

for exp_name, out_dir, src, tgt, to_target in experiments:
    print(f"\n\n######## {exp_name} ########")

    costs = {
        bb: {
            "centroid": cde.CentroidDistanceEstimator(),
            "mean": mce.MeanCostEstimator(),
            "gaussian_mmd": gmmde.GaussianMMDEstimator(),
            f"sinkhorn_{bb}_cosine": sde.SinkhornDistanceEstimator("cosine", bb),
            f"sinkhorn_{bb}_sqeuclidean": sde.SinkhornDistanceEstimator(
                "sqeuclidean", bb
            ),
        }
        for bb in BACKBONES
    }

    for fold in FOLDS:
        fold_dir = out_dir / fold
        print(f"\n=== {fold} ===")

        inp = [Path(p) for p in (fold_dir / "val_files.txt").read_text().splitlines()]
        target = [to_target(p) for p in inp]

        for bb in BACKBONES:
            save_dir = fold_dir / "val_costs" / bb
            first_file = save_dir / f"sinkhorn_{bb}_cosine_mean.csv"

            if first_file.exists():
                print(f"  [{bb}] already done")
                continue

            print(f"  [{bb}] extracting features...")
            inp_feat = extractors[bb].extract(inp)
            target_feat = extractors[bb].extract(target)

            draws = build_bootstrap_draws(
                {src: inp_feat, tgt: target_feat},
                n_repeats=N_REPEATS,
                n_bootstrap=N_BOOTSTRAP,
            )

            save_dir.mkdir(parents=True, exist_ok=True)

            for name, est in costs[bb].items():
                mean, std = est.compute_cost_matrix(draws)
                domains = draws.domain_names

                pd.DataFrame(mean, index=domains, columns=domains).to_csv(
                    save_dir / f"{name}_mean.csv"
                )
                pd.DataFrame(std, index=domains, columns=domains).to_csv(
                    save_dir / f"{name}_std.csv"
                )

            print(f"  [{bb}] saved")

    rows = []

    for fold in FOLDS:
        fold_dir = out_dir / fold
        row = {"fold": fold}

        for bb in BACKBONES:
            names = [
                "centroid",
                "mean",
                "gaussian_mmd",
                f"sinkhorn_{bb}_cosine",
                f"sinkhorn_{bb}_sqeuclidean",
            ]

            for name in names:
                path = fold_dir / "val_costs" / bb / f"{name}_mean.csv"
                df = pd.read_csv(path, index_col=0)
                row[f"{bb}/{name}"] = float(df.iloc[0, 1])

        metrics_df = pd.read_csv(fold_dir / "metrics.csv")
        test_row = metrics_df[metrics_df["Epoch"] == "test"].iloc[0]

        for m in METRICS:
            row[m] = float(test_row[m])

        rows.append(row)

    df = pd.DataFrame(rows).set_index("fold")

    corrs = []

    for cost in [c for c in df.columns if "/" in c]:
        for metric in METRICS:
            x = df[cost].values
            y = df[metric].values

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sr, sp = spearmanr(x, y)

            pr, pp = pearsonr(x, y)

            corrs.append(
                {
                    "experiment": exp_name,
                    "cost": cost,
                    "metric": metric,
                    "pearson_r": round(float(pr), 4),
                    "pearson_p": round(float(pp), 4),
                    "spearman_r": round(float(sr), 4),
                    "spearman_p": round(float(sp), 4),
                    "n": len(FOLDS),
                }
            )

    corr_df = pd.DataFrame(corrs)
    out_file = out_dir / "val_cost_correlation.csv"
    corr_df.to_csv(out_file, index=False)

    print(f"\nCorrelations saved to {out_file}")
