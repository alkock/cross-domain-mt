"""
This file consists of the code for the HE to IHC experiment.
The experiment is conducted on the full dataset using the pre-provided
train/val/test split of HER2Match.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from torch.utils.data import DataLoader

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
        "input_dir": "/workspace/mnt-data/HE/train",
        "target_dir": "/workspace/mnt-data/IHC/train",
    },
    "val": {
        "input_dir": "/workspace/mnt-data/HE/val",
        "target_dir": "/workspace/mnt-data/IHC/val",
    },
    "test": {
        "input_dir": "/workspace/mnt-data/HE/test",
        "target_dir": "/workspace/mnt-data/IHC/test",
    },
}

EPOCHS = 50
L1_LAMBDA = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.0002
NUM_WORKERS = 8
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "heihc_full"

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


def compute_and_save_costs(
    extractor: FeatureExtractor,
    he_paths: list,
    ihc_paths: list,
    output_dir: Path,
    model_name: str,
) -> None:
    print(
        f"  [{model_name}] Extracting features for {len(he_paths)} HE / {len(ihc_paths)} IHC images..."
    )
    he_feats = extractor.extract(he_paths)
    ihc_feats = extractor.extract(ihc_paths)

    draws = build_bootstrap_draws(
        {"HE": he_feats, "IHC": ihc_feats},
        n_repeats=N_REPEATS,
        n_bootstrap=N_BOOTSTRAP,
    )

    costs_dir = output_dir / "pre_costs" / model_name
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
    print(f"  [{model_name}] Pre-costs saved to {costs_dir}")


def main():
    torch.backends.cudnn.benchmark = True
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = PairedImageDataset(
        DATASET["train"]["input_dir"],
        DATASET["train"]["target_dir"],
    )
    val_ds = PairedImageDataset(
        DATASET["val"]["input_dir"],
        DATASET["val"]["target_dir"],
    )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Pre-cost calculation: feature-space distance between HE and IHC (train split)
    train_ihc_paths = [train_ds.target_dir / p.name for p in train_ds.input_paths]
    extractors = {m: FeatureExtractor(model_name=m) for m in FEATURE_MODELS}
    for model_name, extractor in extractors.items():
        compute_and_save_costs(
            extractor, train_ds.input_paths, train_ihc_paths, OUTPUT_DIR, model_name
        )

    # Saving the paths for later analysis
    (OUTPUT_DIR / "train_files.txt").write_text(
        "\n".join(str(p) for p in train_ds.input_paths)
    )
    (OUTPUT_DIR / "val_files.txt").write_text(
        "\n".join(str(p) for p in val_ds.input_paths)
    )

    Pix2PixTrainer(
        dataset={"train_loader": train_loader, "val_loader": val_loader},
        output_dir=str(OUTPUT_DIR),
        epochs=EPOCHS,
        l1_lambda=L1_LAMBDA,
        lr=LEARNING_RATE,
        eval_every=1,
        image_saver=ImageSaver(output_dir=str(OUTPUT_DIR), n_samples=8, save_every=5),
    ).run()

    # Test evaluation
    test_ds = PairedImageDataset(
        DATASET["test"]["input_dir"],
        DATASET["test"]["target_dir"],
    )
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    Pix2PixTrainer(
        output_dir=str(OUTPUT_DIR),
        image_saver=ImageSaver(output_dir=str(OUTPUT_DIR), n_samples=8),
    ).evaluate(test_loader)


if __name__ == "__main__":
    main()
