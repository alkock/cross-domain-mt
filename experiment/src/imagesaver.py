import numpy as np
import torch
from pathlib import Path
from PIL import Image


class ImageSaver:
    def __init__(self, output_dir: str, n_samples: int = 8, save_every: int = 5):
        self.output_dir = Path(output_dir)
        self.n_samples = n_samples
        self.save_every = save_every

    def _to_uint8(self, t: torch.Tensor) -> np.ndarray:
        arr = (t * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return (arr * 255).astype(np.uint8)

    def save(self, gen: torch.nn.Module, loader, tag: str, device: str) -> None:
        folder = self.output_dir / "images" / tag
        folder.mkdir(parents=True, exist_ok=True)

        gen.eval()
        saved = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                fake = gen(x)
                for i in range(x.size(0)):
                    if saved >= self.n_samples:
                        break
                    grid = np.concatenate(
                        [
                            self._to_uint8(x[i]),
                            self._to_uint8(fake[i]),
                            self._to_uint8(y[i]),
                        ],
                        axis=1,
                    )
                    Image.fromarray(grid).save(folder / f"{saved:03d}.png")
                    saved += 1
                if saved >= self.n_samples:
                    break
