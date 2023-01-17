import os
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int, dirpath: str):
        super().__init__()
        self.every = every
        self.dirpath = dirpath

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

    def on_before_zero_grad(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.global_step % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"latest-{pl_module.global_step}.ckpt"
            prev = (
                Path(self.dirpath) / f"latest-{pl_module.global_step - self.every}.ckpt"
            )
            print(current)
            trainer.save_checkpoint(current)

