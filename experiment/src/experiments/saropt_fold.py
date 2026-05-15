"""
This file consists of the code for the SAR to OPT experiment.
The experiment is conducted on a created split of the experiment.
The split is created with the sen12_split.ipynb file,
located in experiment/preprocessing/sen12_split.ipynb.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import GroupKFold
from torch.utils.data import ConcatDataset, DataLoader, Subset

import experiment.src.cost_modules.centroid_distance_estimator as cde
import experiment.src.cost_modules.gaussian_mmd_estimator as gmmde
import experiment.src.cost_modules.mean_cost_estimator as mce
import experiment.src.cost_modules.sinkhorn_distance_estimator as sde
from experiment.src.dataset import PairedImageDataset
from experiment.src.feature_extractor import FeatureExtractor, build_bootstrap_draws
from experiment.src.imagesaver import ImageSaver
from experiment.src.trainer import Pix2PixTrainer

DATASET = {
    "train": {
        "input_dir": "/workspace/mnt-data/SEN12-splits/train/input",
        "target_dir": "/workspace/mnt-data/SEN12-splits/train/target",
    },
    "val": {
        "input_dir": "/workspace/mnt-data/SEN12-splits/val/input",
        "target_dir": "/workspace/mnt-data/SEN12-splits/val/target",
    },
    "test": {
        "input_dir": "/workspace/mnt-data/SEN12-splits/test/input",
        "target_dir": "/workspace/mnt-data/SEN12-splits/test/target",
    },
}

N_FOLDS = 12
EPOCHS = 50
L1_LAMBDA = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.0002
NUM_WORKERS = 8
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "saropt_fold"
N_REPEATS = 50
N_BOOTSTRAP = 389
FEATURE_MODELS = ["resnet18", "resnet50", "dinov2", "vgg16"]

COST_ESTIMATORS = {
    backbone: {
        "centroid": cde.CentroidDistanceEstimator(),
        "mean": mce.MeanCostEstimator(),
        "gaussian_mmd": gmmde.GaussianMMDEstimator(),
        f"sinkhorn_{backbone}_cosine": sde.SinkhornDistanceEstimator(
            "cosine", backbone
        ),
        f"sinkhorn_{backbone}_sqeuclidean": sde.SinkhornDistanceEstimator(
            "sqeuclidean", backbone
        ),
    }
    for backbone in FEATURE_MODELS
}


def compute_and_save_fold_costs(
    extractor: FeatureExtractor,
    sar_paths: list,
    opt_paths: list,
    fold_dir: Path,
    model_name: str,
) -> None:
    print(
        f"  [{model_name}] Extracting features for {len(sar_paths)} SAR / {len(opt_paths)} OPT images..."
    )
    sar_feats = extractor.extract(sar_paths)
    opt_feats = extractor.extract(opt_paths)

    draws = build_bootstrap_draws(
        {"SAR": sar_feats, "OPT": opt_feats},
        n_repeats=N_REPEATS,
        n_bootstrap=N_BOOTSTRAP,
    )

    costs_dir = fold_dir / "pre_costs" / model_name
    costs_dir.mkdir(parents=True, exist_ok=True)

    for name, estimator in COST_ESTIMATORS[model_name].items():
        mean_mat, std_mat = estimator.compute_cost_matrix(draws)
        domain_names = draws.domain_names
        pd.DataFrame(mean_mat, index=domain_names, columns=domain_names).to_csv(
            costs_dir / f"{name}_mean.csv"
        )
        pd.DataFrame(std_mat, index=domain_names, columns=domain_names).to_csv(
            costs_dir / f"{name}_std.csv"
        )
    print(f"  [{model_name}] Fold costs saved to {costs_dir}")


def main():
    torch.backends.cudnn.benchmark = True
    train_ds = PairedImageDataset(
        DATASET["train"]["input_dir"],
        DATASET["train"]["target_dir"],
    )

    val_ds = PairedImageDataset(
        DATASET["val"]["input_dir"],
        DATASET["val"]["target_dir"],
    )

    # Combine train and val for GroupKFold (test split stays separate)
    combined_ds = ConcatDataset([train_ds, val_ds])
    all_paths = train_ds.input_paths + val_ds.input_paths
    all_opt_paths = [train_ds.target_dir / p.name for p in train_ds.input_paths] + [
        val_ds.target_dir / p.name for p in val_ds.input_paths
    ]
    # Scene-ID = filename prefix before "_pNNN"
    groups = [p.stem.rsplit("_p", 1)[0] for p in all_paths]

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
    }
    gkf = GroupKFold(n_splits=N_FOLDS)
    extractors = {m: FeatureExtractor(model_name=m) for m in FEATURE_MODELS}

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(range(len(all_paths)), groups=groups)
    ):
        fold_dir = OUTPUT_DIR / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_paths = [all_paths[i] for i in train_idx]
        val_paths = [all_paths[i] for i in val_idx]
        train_opt_paths = [all_opt_paths[i] for i in train_idx]

        # Save explicit filenames per fold
        (fold_dir / "train_files.txt").write_text(
            "\n".join(str(p) for p in train_paths)
        )
        (fold_dir / "val_files.txt").write_text("\n".join(str(p) for p in val_paths))

        # Fold cost calculation: feature-space distance between SAR and OPT (train split)
        for model_name, extractor in extractors.items():
            compute_and_save_fold_costs(
                extractor, train_paths, train_opt_paths, fold_dir, model_name
            )

        train_loader = DataLoader(
            Subset(combined_ds, train_idx), shuffle=True, **loader_kwargs
        )
        val_loader = DataLoader(
            Subset(combined_ds, val_idx), shuffle=False, **loader_kwargs
        )

        Pix2PixTrainer(
            dataset={"train_loader": train_loader, "val_loader": val_loader},
            output_dir=str(fold_dir),
            epochs=EPOCHS,
            l1_lambda=L1_LAMBDA,
            lr=LEARNING_RATE,
            eval_every=1,
            image_saver=ImageSaver(output_dir=str(fold_dir), n_samples=8, save_every=5),
        ).run()

    # Test evaluation
    test_ds = PairedImageDataset(
        DATASET["test"]["input_dir"],
        DATASET["test"]["target_dir"],
    )
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    for fold in range(N_FOLDS):
        fold_dir = OUTPUT_DIR / f"fold_{fold:02d}"
        Pix2PixTrainer(
            output_dir=str(fold_dir),
            image_saver=ImageSaver(output_dir=str(fold_dir), n_samples=8),
        ).evaluate(test_loader)


if __name__ == "__main__":
    main()
