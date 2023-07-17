import os
import random

import backbone
import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from filelock import FileLock
from torchmetrics import MetricCollection
from torchmetrics.classification import F1Score, Recall


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
        self.save_hyperparameters()
        # Create model
        self.backbone = getattr(backbone, kwargs.get("bb_name", "ResNet"))(num_classes, **(kwargs.get("bb_hparams", {})))
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), device=self.device)
        # Creates metrics that would be logged
        metric = MetricCollection(
            Recall(task="multiclass", num_classes=num_classes, average="weighted"),
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
        # metric[f"{stage}_MulticlassRecall"] = torch.exp(torch.mean(torch.log(metric[f"{stage}_MulticlassRecall"])))
        self.log_dict(metric, prog_bar=True)

        if stage == "train":
            loss = self.loss_module(logits, labels)
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            return loss
        
        # Create pandas dataframe if stage is not train
        df = pd.DataFrame(logits.cpu().numpy()).add_prefix("Logit_")
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

class M2mModule(ERMModule):
    def __init__(self, m2m_epoch, rej_prob, attack_iters, regul_param, step_size, misclass_bound, checkpoint_path, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def generate(self, batch):
        # The models are in eval mode, 
        # it's the input that is going to be optimized
        self.freeze()

        imgs, labels = [item.clone() for item in batch]
        # Number of training samples per class
        # This is on cpu by default
        counts = self.trainer.datamodule.class_counts.type_as(labels)
        # Create a binary mask for classes present in the batch
        in_batch = torch.zeros_like(counts)
        # Create a dict of imgs grouped according to their class labels
        img_dict = {}
        for x, y in zip(imgs, labels):
            img_dict.setdefault(y.item(), []).append(x.clone())
            in_batch[y] = 1
        
        # Initialize a list of masks
        masks = []
        
        # The imbalance ratio of the corresponding class
        imb_ratio = counts[labels] / torch.max(counts)
        # Mask 1: Use imb_ratio to reject items
        masks.append(torch.bernoulli(1 - imb_ratio).bool())
        y_minor = labels[masks[-1]]
         
        # If the intended source class has same or less number of samples 
        # source_reject completely. Zero those classes that are absent.
        source_reject = self.hparams.rej_prob ** torch.relu(counts * in_batch - counts[y_minor].unsqueeze(1))
        # Mask 2: Check if the source probs add up to a positive number
        masks.append((1 - source_reject).sum(dim=1) > 0)
        source_reject = source_reject[masks[-1]]
        y_minor = y_minor[masks[-1]]

        # If not, randomly select the intended source 
        # class, using this probability.
        y_major = torch.multinomial(input=(1 - source_reject), num_samples=1).squeeze()
        # Mask 3: Filter using source_reject
        masks.append(~torch.bernoulli(source_reject[torch.arange(len(y_major)), y_major]).bool())
        y_major = y_major[masks[-1]]
        y_minor = y_minor[masks[-1]]

        # Randomly selecting an image given the source classes
        x_major = torch.stack([random.choice(img_dict[y.item()]) for y in y_major])
        # Load oracle to compute gradients
        oracle = ERMModule.load_from_checkpoint(**self.hparams, map_location=self.device)
        oracle.freeze()
        # Preparing the soucre image for translating into 
        # the labels class. 
        x = x_major.requires_grad_()
        # Small initial perturbation to the input image
        x = torch.clamp(x + torch.rand_like(x), -1, 1)
        for _ in range(self.hparams.attack_iters):
            # Loss that is minized plus a regularizer term
            loss = oracle.loss_module(oracle(x), y_minor) + self.hparams.regul_param * self(x)[torch.arange(len(y_major)), y_major].mean()
            # Do manual backward pass according to rules of Pytorch Lightning
            grad = torch.autograd.grad(loss, x)[0]
            grad /= grad.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-8
            # Update the inputs
            x = x - self.hparams.step_size * grad
            x = torch.clamp(x, -1, 1)
            
        x_minor = x.detach()
        # Mask 4: Accept the generated image by a loss threshold and random rejection
        masks.append(nn.CrossEntropyLoss(reduction="none")(oracle(x_minor), y_minor) < self.hparams.misclass_bound)
        x_minor = x_minor[masks[-1]]

        # find originial indices of each item in x_minor
        indices = torch.arange(masks[-1].sum())
        for mask in reversed(masks):
            non_zero = mask.nonzero()[indices]
            indices = non_zero.squeeze()
        
        # Update the images
        imgs[indices] = x_minor

        self.unfreeze()

        return [imgs, labels]

    def training_step(self, batch, batch_idx):
        return super().training_step(batch if self.current_epoch < self.hparams.m2m_epoch else self.generate(batch), batch_idx)