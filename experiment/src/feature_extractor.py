import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from experiment.src.models import BootstrapDraws, DomainFeatures, Domain


class FeatureExtractor:
    def __init__(self, model_name="resnet50", batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        self.model, self.preprocess = self._build(model_name)
        self.model.to(self.device).eval()

    def _build(self, name):
        if name == "resnet50":
            backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
            model = nn.Sequential(*list(backbone.children())[:-1])
        elif name == "resnet18":
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
            model = nn.Sequential(*list(backbone.children())[:-1])
        elif name == "vgg16":
            backbone = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
            model = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d((1, 1)))
        elif name == "dinov2":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        else:
            raise ValueError(f"Unknown model: {name}")
        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return model, preprocess

    def extract(self, paths: list) -> np.ndarray:
        """Extract features for a list of image paths. Returns (N, D) array."""
        ds = _PathDataset(paths, self.preprocess)
        loader = DataLoader(
            ds, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )
        batches = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                if self.model_name == "dinov2":
                    out = self.model.forward_features(batch)
                    feats = out["x_norm_clstoken"]
                else:
                    feats = self.model(batch).flatten(1)
                batches.append(feats.cpu().numpy())
        return np.concatenate(batches, axis=0)


class _PathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_bootstrap_draws(
    domain_features: dict,
    n_repeats: int = 50,
    n_bootstrap: int = 389,
    seed: int = 0,
) -> BootstrapDraws:
    """
    Args:
        domain_features: {"domain_name": np.ndarray of shape (N, D)}
        n_repeats: number of bootstrap repetitions
        n_bootstrap: samples per repeat (capped at domain size)
        seed: random seed
    """
    rng = np.random.default_rng(seed)
    domain_feature_list = []
    repeats = {}

    for name, feats in domain_features.items():
        domain = Domain(name=name, path="")
        domain_feature_list.append(DomainFeatures(domain=domain, features=feats))
        n = feats.shape[0]
        k = min(n_bootstrap, n)
        repeats[name] = [
            feats[rng.choice(n, k, replace=True)] for _ in range(n_repeats)
        ]

    return BootstrapDraws(
        domain_features=domain_feature_list,
        repeats=repeats,
        n_repeats=n_repeats,
        n_bootstrap=n_bootstrap,
    )
