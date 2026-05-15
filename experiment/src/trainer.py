import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from experiment.src.pix2pix.generator import Generator
from experiment.src.pix2pix.discriminator import Discriminator
from experiment.src.dataset import PairedImageDataset
from experiment.src.metricslogger import MetricsLogger
from experiment.src.imagesaver import ImageSaver
from experiment.src.lossplotter import LossPlotter


class Pix2PixTrainer:
    def __init__(
        self,
        dataset=None,
        output_dir="output",
        epochs=50,
        l1_lambda=100,
        batch_size=8,
        lr=0.0002,
        num_workers=4,
        metrics=("psnr", "ssim", "lpips", "fid", "kid"),
        eval_every=1,
        image_saver: ImageSaver | None = None,
    ):
        self.epochs = epochs
        self.l1_lambda = l1_lambda
        self.eval_every = eval_every
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if dataset is not None:
            self._build_dataloaders(dataset, batch_size, num_workers)
        self._build_models()
        self._build_optimizers(lr)
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.scaler = torch.amp.GradScaler()
        self.metrics_logger = MetricsLogger(
            output_dir=str(self.output_dir),
            device=self.device,
            metrics=metrics,
        )
        self.image_saver = image_saver

    def _build_dataloaders(self, dataset, batch_size, num_workers):
        if "train_loader" in dataset:
            self.train_loader = dataset["train_loader"]
            self.val_loader = dataset["val_loader"]
        else:
            train_ds = PairedImageDataset(
                dataset["train"]["input_dir"],
                dataset["train"]["target_dir"],
            )
            val_ds = PairedImageDataset(
                dataset["val"]["input_dir"],
                dataset["val"]["target_dir"],
            )
            self.train_loader = DataLoader(
                train_ds,
                shuffle=True,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )
            self.val_loader = DataLoader(
                val_ds,
                shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )
        self.metrics_loader = DataLoader(
            self.val_loader.dataset,
            shuffle=False,
            batch_size=64,
            num_workers=self.val_loader.num_workers,
            pin_memory=True,
        )

    def _build_models(self):
        self.gen = Generator(in_channels=3, out_channels=3).to(self.device)
        self.disc = Discriminator(in_channels=3).to(self.device)

    def _build_optimizers(self, lr):
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=lr, betas=(0.5, 0.999))

    def _train_epoch(self):
        self.gen.train()
        self.disc.train()
        total_l1 = 0.0

        for x, y in tqdm(self.train_loader, leave=False):
            x, y = x.to(self.device), y.to(self.device)

            with torch.amp.autocast(device_type="cuda"):
                y_fake = self.gen(x)
                d_real = self.disc(x, y)
                d_fake = self.disc(x, y_fake.detach())
                d_loss = (
                    self.bce(d_real, torch.ones_like(d_real))
                    + self.bce(d_fake, torch.zeros_like(d_fake))
                ) / 2

            self.opt_disc.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.opt_disc)

            with torch.amp.autocast(device_type="cuda"):
                d_fake = self.disc(x, y_fake)
                l1_loss = self.l1(y_fake, y)
                g_loss = (
                    self.bce(d_fake, torch.ones_like(d_fake)) + self.l1_lambda * l1_loss
                )

            self.opt_gen.zero_grad()
            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.opt_gen)
            self.scaler.update()

            total_l1 += l1_loss.item()

        return total_l1 / len(self.train_loader)

    @torch.no_grad()
    def _eval_metrics(self, epoch: int, train_l1: float) -> tuple[float, dict]:
        self.gen.eval()
        metrics = self.metrics_logger.compute_with_l1(self.gen, self.metrics_loader)
        val_l1 = metrics.pop("l1")

        self.metrics_logger.log(epoch, train_l1, val_l1, metrics)
        if self.image_saver is not None and epoch % self.image_saver.save_every == 0:
            self.image_saver.save(
                self.gen, self.val_loader, f"epoch_{epoch:03d}", self.device
            )
        self.gen.train()
        return val_l1, metrics

    def _save_checkpoint(self, epoch, val_l1):
        torch.save(
            {
                "epoch": epoch,
                "gen_state_dict": self.gen.state_dict(),
                "disc_state_dict": self.disc.state_dict(),
                "opt_gen_state_dict": self.opt_gen.state_dict(),
                "opt_disc_state_dict": self.opt_disc.state_dict(),
                "best_val_l1": val_l1,
            },
            self.output_dir / "best.pt",
        )

    def run(self):
        print(f"Using device: {self.device}")
        best_val_l1 = float("inf")
        history = []

        for epoch in range(1, self.epochs + 1):
            train_l1 = self._train_epoch()
            val_l1, metrics = self._eval_metrics(epoch, train_l1)

            history.append(
                {"epoch": epoch, "train_l1": train_l1, "val_l1": val_l1, **metrics}
            )
            print(
                f"Epoch [{epoch:3d}/{self.epochs}]  train_L1={train_l1:.4f}  val_L1={val_l1:.4f}"
                + (
                    f"  PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}"
                    if metrics
                    else ""
                )
            )

            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                self._save_checkpoint(epoch, val_l1)
                print(f"  -> Saved best checkpoint (val_L1={best_val_l1:.4f})")

        LossPlotter(self.output_dir).save(history)
        return history

    def evaluate(self, test_loader) -> dict:
        checkpoint = self.output_dir / "best.pt"
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=self.device)
            self.gen.load_state_dict(state["gen_state_dict"])
            print(
                f"Loaded checkpoint from epoch {state['epoch']} (val_L1={state['best_val_l1']:.4f})"
            )

        self.gen.eval()
        metrics_loader = DataLoader(
            test_loader.dataset,
            shuffle=False,
            batch_size=64,
            num_workers=test_loader.num_workers,
            pin_memory=True,
        )
        metrics = self.metrics_logger.compute_with_l1(self.gen, metrics_loader)
        test_l1 = metrics.pop("l1")
        self.metrics_logger.log("test", None, test_l1, metrics)

        if self.image_saver is not None:
            self.image_saver.save(self.gen, test_loader, "test", self.device)

        print(
            f"Test  L1={test_l1:.4f}"
            + (
                f"  PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}"
                if metrics
                else ""
            )
        )
        return {"test_l1": test_l1, **metrics}
