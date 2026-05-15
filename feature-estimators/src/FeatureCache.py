import os
import numpy as np


class FeatureCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def path(self, dataset_name, model_name, pool_size, seed):
        filename = f"{dataset_name}_{model_name}_pool{pool_size}_seed{seed}.npy"
        return os.path.join(self.cache_dir, filename)

    def load(self, path):
        if not os.path.exists(path):
            return None
        arr = np.load(path)
        print(f"  -> cache hit ({len(arr)} features)")
        return arr

    def save(self, arr, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, arr)
        print(f"  -> cached to {path}")
