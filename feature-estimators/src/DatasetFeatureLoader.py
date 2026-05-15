import numpy as np
import torch
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from ImageFileDataset import ImageFileDataset
import os

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


class DatasetFeatureLoader:
    def __init__(self, extractor, pool_size=None, seed=0, batch_size=64, cache=None):
        self.extractor = extractor
        self.pool_size = pool_size
        self.seed = seed
        self.batch_size = batch_size
        self.cache = cache

    def load(self, folder_path, dataset_name):
        if self.cache:
            cache_path = self.cache.path(
                dataset_name, self.extractor.model_name, self.pool_size, self.seed
            )
            cached = self.cache.load(cache_path)
            if cached is not None:
                return cached

        files = self._collect_files(folder_path)
        loader = self._build_dataloader(folder_path, files)
        batches = [self._extract_batch(batch) for batch in loader]
        result = np.concatenate(batches, axis=0) if batches else np.empty((0,))

        if self.cache:
            self.cache.save(result, cache_path)

        return result

    def _collect_files(self, folder_path):
        files = sorted(
            f
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTENSIONS
        )
        if self.pool_size and len(files) > self.pool_size:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(files), self.pool_size, replace=False)
            files = [files[i] for i in sorted(idx)]
        return files

    def _build_dataloader(self, folder_path, files):
        ds = ImageFileDataset(folder_path, files, self.extractor.preprocess)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

    def _extract_batch(self, batch):
        batch = batch.to(self.extractor.device)
        with torch.no_grad():
            if self.extractor.model_name == "dinov2":
                out = self.extractor.model.forward_features(batch)
                return out["x_norm_clstoken"].detach().cpu().numpy()
            return self.extractor.model(batch).detach().cpu().flatten(1).numpy()
