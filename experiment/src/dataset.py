import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class PairedImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, preload=False):
        self.input_paths = sorted(Path(input_dir).glob("*"))
        self.target_dir = Path(target_dir)

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self._inputs = None
        self._targets = None
        if preload:
            self._preload()

    def _preload(self):
        print(f"Preloading {len(self.input_paths)} image pairs into RAM...")
        self._inputs = []
        self._targets = []
        for p in tqdm(self.input_paths, leave=False):
            self._inputs.append(self.transform(Image.open(p).convert("RGB")))
            self._targets.append(
                self.transform(Image.open(self.target_dir / p.name).convert("RGB"))
            )
        self._inputs = torch.stack(self._inputs)
        self._targets = torch.stack(self._targets)
        print(
            f"  Done. Memory: {self._inputs.nbytes / 1e9:.1f} GB input + {self._targets.nbytes / 1e9:.1f} GB target"
        )

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        if self._inputs is not None:
            return self._inputs[idx], self._targets[idx]
        inp_path = self.input_paths[idx]
        tgt_path = self.target_dir / inp_path.name
        return (
            self.transform(Image.open(inp_path).convert("RGB")),
            self.transform(Image.open(tgt_path).convert("RGB")),
        )
