import os
import random
from typing import Any, Optional

from lightning.pytorch.utilities.types import STEP_OUTPUT

import backbone
import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from filelock import FileLock
from torchmetrics import MetricCollection
from torchmetrics.classification import F1Score, Recall
from torch.optim import SGD

class ERMModule(L.LightningModule):
    def __init__(self, num_classes, **kwargs):
        """ERMModule.

        Args:
            backbone_name: Name of the model/CNN to run. Used for creating the model (see function below)
            backbone_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        # Create model
        self.backbone = getattr(backbone, kwargs.get("bb_name", "ResNet"))(num_classes, **(kwargs.get("bb_hparams", {})))
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), device=self.device)
        # Creates metrics that would be logged
        metric = MetricCollection(
            Recall(task="multiclass", num_classes=num_classes, average=None),
            F1Score(task="multiclass", num_classes=num_classes, average="weighted"),
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
        # Obatin dict of metrics
        metric = getattr(self, f"{stage}_metrics")(logits, labels)
        # Compute geometric mean score
        metric[f"{stage}_MulticlassRecall"] = torch.exp(torch.mean(torch.log(metric[f"{stage}_MulticlassRecall"])))
        self.log_dict(metric, sync_dist=True, prog_bar=True)

        if stage == "train":
            loss = self.loss_module(logits, labels)
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            return loss
        
        # Create pandas dataframe if stage is not train
        df = pd.DataFrame(logits.cpu().numpy()).add_prefix("Prob_")
        df["True"] = labels.cpu()
        # Save labels and predictions to a file with file locking
        filename = f"{self.trainer.log_dir}/step_{self.global_step:06}_{stage}_labels_logits.csv"
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

class M2mModule(L.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.g = ERMModule.load_from_checkpoint("../results/erm_sun397/lightning_logs/version_20/checkpoints/epoch=89-step=31320.ckpt").backbone
        self.f = ERMModule(num_classes=self.g.hparams.num_classes)
        self.beta = 0.9
        self.T = 10
        self.lam = 0.5
        self.gamma = 0.9
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # Number of training samples per class
        counts = self.trainer.datamodule.class_counts
        
        x0, y0 = batch
        imgs = x0.clone()
        labels = y0.clone()
        
        # Create a binary mask for classes present in the batch
        in_batch = torch.zeros_like(counts)
        in_batch[labels] = 1
        # Create a dict of imgs grouped according to their class labels
        img_dict = {}
        for x, y in zip(imgs, labels):
            img_dict.setdefault(y, []).append(x)

        # Decide if we would generate an image for a labels class
        # using the imbalance ratio.
        gen = torch.bernoulli(1 - counts[labels] / torch.max(counts)).bool()
        if torch.rand(1).item() < :
            # Conditional probability of rejecting the image. If
            # the source class has same or less number of samples 
            # reject completely.
            probs = self.beta ** torch.maximum(counts - counts[labels], torch.zeros_like(counts))
            # Randomly select the source class, ensuring that the
            # source class is actually in the batch
            source = torch.multinomial((1 - probs) * in_batch, 1)
            # Randomly selecting an image of the source class
            seed = random.choice(img_dict[source])
            # Preparing the soucre image for translating into 
            # the labels class
            x = seed.clone().requires_grad_(True)
            # The models are in eval mode, it's the input that is going to be optimized
            optimizer = SGD([x], lr=0.1)
            # Small initial perturbation to the input image
            x = torch.clamp(x + torch.randn_like(x), -1, 1)
            # Need to track this to decide if the generated image ought to be rejected
            loss_g = 0
            for _ in range(self.T):
                loss_g = self.g.loss_module(self.g(x), labels)
                # Loss that is minized plus a regularizer term
                loss = loss_g + self.lam * self.f(x)[source]
                # Normalized loss
                loss = torch.div(loss, torch.norm(loss))
                # Propagating the loss appropriately
                optimizer.zero_grad()
                loss.backward()
                # Update the seed image
                optimizer.step()
                x.data = torch.clamp_(x.data, -1, 1)

            # Reject the generated image or randomly choose an unchanged image belonging to the labels class
            if loss_g > self.gamma:
                imgs[labels] = x.detach()

        # With the modified images, train the actual submodule
        logits = self.f(imgs)
        loss = self.f.loss_module(logits, labels)
        return loss