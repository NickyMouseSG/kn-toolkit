import pytorch_lightning as pl


class CaptureOutput(pl.Callback):
    # https://github.com/PyTorchLightning/pytorch-lightning/discussions/11659
    def __init__(self) -> None:
        self.ins = []
        self.outs = []

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.ins = []
        self.outs = []

    def on_test_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.ins = []
        self.outs = []

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.ins.append(batch)
        self.outs.append(outputs)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ):
        self.ins.append(batch)
        self.out.append(outputs)
