import os

import backbone
import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from filelock import FileLock
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassRecall


class ERMModule(L.LightningModule):
    def __init__(
            self, backbone_name="ResNet", 
            backbone_hparams={
                "num_classes": None, 
                "num_blocks": [2, 2, 2, 2], 
                "c_hidden": [64, 128, 256, 512],
            }, 
            # optimizer_name="SGD", 
            # optimizer_hparams={
            #     "lr": 0.1, "momentum": 0.9, "weight_decay": 2e-4
            # }
    ):
        """ERMModule.

        Args:
            backbone_name: Name of the model/CNN to run. Used for creating the model (see function below)
            backbone_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.backbone = getattr(backbone, backbone_name)(**backbone_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32))
        # Creates metrics that would be logged
        metric = MetricCollection(
            MulticlassRecall(num_classes=self.backbone.hparams.num_classes, average=None),
            MulticlassF1Score(num_classes=self.backbone.hparams.num_classes, average="weighted"),
        )
        self.train_metrics = metric.clone(prefix="train_")
        self.val_metrics = metric.clone(prefix="val_")
        self.test_metrics = metric.clone(prefix="test_")

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.backbone(imgs)

   
    # Avoid duplication of code in the step methods
    def _unified_step(self, batch, stage: str):
        imgs, labels = batch
        logits = self(imgs)
        preds = logits.softmax(dim=1)
        # Obatin dict of metrics
        metric = getattr(self, f"{stage}_metrics")(preds, labels)
        # Compute geometric mean score
        metric[f"{stage}_MulticlassRecall"] = torch.exp(torch.mean(torch.log(metric[f"{stage}_MulticlassRecall"])))
        self.log_dict(metric, on_epoch=True, sync_dist=True)

        if stage == "train":
            loss = self.loss_module(logits, labels)
            self.log("train_loss", loss)
            return loss
        
        # Create pandas dataframe if stage is not train
        df = pd.DataFrame(preds.cpu().numpy()).add_prefix("Prob_")
        df["True"] = labels.cpu()
        # Save labels and predictions to a file with file locking
        filename = f"{self.trainer.log_dir}/{stage}_labels_preds.csv"
        with FileLock(f"{filename}.lock"):
            df.to_csv(filename, mode="a", header=not os.path.isfile(filename), index=False)
        
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        return self._unified_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._unified_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._unified_step(batch, "test")
    
    def on_training_epoch_end(self):
        self.train_metrics.reset()
        
    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.test_metrics.reset()