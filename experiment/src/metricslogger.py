import os
import csv
from typing import Sequence

import torch
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

_COLS = {
    "psnr": ["PSNR"],
    "ssim": ["SSIM"],
    "lpips": ["LPIPS"],
    "fid": ["FID"],
    "kid": ["KID_Mean", "KID_Std"],
}


class MetricsLogger:
    def __init__(
        self,
        output_dir: str,
        filename: str = "metrics.csv",
        device: str = "cuda",
        metrics: Sequence[str] = ("psnr", "ssim", "lpips", "fid", "kid"),
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, filename)
        self.device = device
        self.metrics = tuple(metrics)

        self.lpips_fn = (
            lpips.LPIPS(net="alex").to(device).eval() if "lpips" in metrics else None
        )
        self.psnr = (
            PeakSignalNoiseRatio(data_range=1.0).to(device)
            if "psnr" in metrics
            else None
        )
        self.ssim = (
            StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            if "ssim" in metrics
            else None
        )
        self.fid = (
            FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            if "fid" in metrics
            else None
        )
        self.kid = (
            KernelInceptionDistance(subset_size=50, normalize=True).to(device)
            if "kid" in metrics
            else None
        )

        if not (os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0):
            headers = ["Epoch", "Train_L1", "Val_L1"] + [
                c for m in metrics for c in _COLS[m]
            ]
            with open(self.filepath, "w", newline="") as f:
                csv.writer(f).writerow(headers)

    def _norm(self, t: torch.Tensor) -> torch.Tensor:
        return (t * 0.5 + 0.5).clamp(0, 1)

    def _to3ch(self, t: torch.Tensor) -> torch.Tensor:
        return t.repeat(1, 3, 1, 1) if t.size(1) == 1 else t

    def compute_with_l1(self, gen: torch.nn.Module, loader) -> dict:
        """Like compute(), but also returns val L1 — avoids a second generator pass."""
        lpips_acc = 0.0
        l1_acc = 0.0
        n_batches = 0

        for tm in (self.psnr, self.ssim, self.fid, self.kid):
            if tm is not None:
                tm.reset()

        l1_fn = torch.nn.L1Loss()

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True
                )
                fake = gen(x)
                y_norm, fake_norm = self._norm(y), self._norm(fake)
                n_batches += 1

                l1_acc += l1_fn(fake, y).item()

                if self.psnr is not None:
                    self.psnr.update(fake_norm, y_norm)

                if self.ssim is not None:
                    self.ssim.update(fake_norm, y_norm)

                if self.lpips_fn is not None:
                    lpips_acc += (
                        self.lpips_fn(self._to3ch(fake), self._to3ch(y)).sum().item()
                    )

                if self.fid is not None or self.kid is not None:
                    real3, fake3 = self._to3ch(y_norm), self._to3ch(fake_norm)
                    for tm in (self.fid, self.kid):
                        if tm is not None:
                            tm.update(real3, real=True)
                            tm.update(fake3, real=False)

        if n_batches == 0:
            return {"l1": float("nan"), **{m: float("nan") for m in self.metrics}}

        result = {"l1": l1_acc / n_batches}
        if self.psnr is not None:
            result["psnr"] = self.psnr.compute().item()
        if self.ssim is not None:
            result["ssim"] = self.ssim.compute().item()
        if self.lpips_fn is not None:
            result["lpips"] = lpips_acc / n_batches
        if self.fid is not None:
            result["fid"] = self.fid.compute().item()
        if self.kid is not None:
            km, ks = self.kid.compute()
            result["kid_mean"], result["kid_std"] = km.item(), ks.item()
        return result

    def log(self, epoch, train_l1: float | None, val_l1: float, metrics: dict):
        train_str = f"{train_l1:.6f}" if train_l1 is not None else ""
        row = [epoch, train_str, f"{val_l1:.6f}"]
        for m in self.metrics:
            keys = ["kid_mean", "kid_std"] if m == "kid" else [m]
            row.extend(f"{metrics[k]:.6f}" for k in keys)
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)
