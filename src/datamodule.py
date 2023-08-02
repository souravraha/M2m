from random import shuffle

import lightning as L
import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets as D
from torchvision import transforms as T


class ImbalancedDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, pin_memory=True):
        super().__init__()
        self.save_hyperparameters()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # Modify the normalization values accordingly
        ])

    def prepare_data(self):
        # Implement dataset downloading or preparation if required
        pass

    def setup(self, stage=None):
        # Initialize datasets here
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    def compute_class_weights(self, dataset):
        # Compute class weights for imbalanced datasets
        if not hasattr(self, "class_counts"):
            self.class_counts = torch.bincount(torch.tensor(dataset.targets))

        weights = self.class_counts.reciprocal()
        return [weights[y] for y in dataset.targets]

    def handle_imbalanced_dataset(self, dataset, apply_class_weights=True):
        if apply_class_weights:
            class_weights = self.compute_class_weights(dataset)
            sampler = WeightedRandomSampler(
                weights=class_weights,
                num_samples=len(dataset),
                replacement=True
            )
            rank_zero_info("Initiating class-balanced sampling...")
            return DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                sampler=sampler
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True
            )

    # Implement any other methods as needed for your specific imbalanced learning tasks
    # For instance, data augmentation methods, custom sampling techniques, etc.

# Define the LightningDataModules
class SUN397DataModule(ImbalancedDataModule):
    name = "sun397"
    MEANS = [0.4833, 0.4656, 0.4285]
    STDS = [0.2646, 0.2618, 0.2847]

    def __init__(self, data_dir, val_split_per_class=10, test_split_per_class=40, oversample_epoch=9999, **kwargs):
        super().__init__(**kwargs)
        # Save init args as hparams
        self.save_hyperparameters()
        # Override the transform attribute
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(
                size=(32, 32), scale=(0.8, 1), ratio=(1, 1), 
                interpolation=T.InterpolationMode.NEAREST
            ),
            T.ToTensor(),
            # The per-channel means and stds are quite robust
            T.Normalize(mean=SUN397DataModule.MEANS, std=SUN397DataModule.STDS),
        ])

    def setup(self, stage=None):
        # Load and split the dataset into train and validation sets
        dataset = D.SUN397(root=self.hparams.data_dir, transform=self.transform)
        # Create a list of indices that belong to each class
        cls_idx_list = [[] for _ in range(len(dataset.classes))]
        # Don't iterate over the dataset as transforming 
        # the images takes up a lot of time
        for i, label in enumerate(dataset._labels):
            cls_idx_list[label].append(i)
        
        # Create val and test splits in a stratified manner
        val_idxs = []
        test_idxs = []
        non_train = self.hparams.val_split_per_class + self.hparams.test_split_per_class
        for i, idx_list in enumerate(cls_idx_list):
            shuffle(idx_list)
            val_idxs.extend(idx_list[ : self.hparams.val_split_per_class])
            test_idxs.extend(idx_list[self.hparams.val_split_per_class : non_train])
            cls_idx_list[i] = len(idx_list) - non_train
        
        # Tensor containing num of samples per class
        self.class_counts = torch.tensor(cls_idx_list)
        # The remaining are the train split
        train_idxs = list(set(range(len(dataset))) - set(val_idxs) - set(test_idxs))
        # Define subsets as required by the stage
        if stage in ["fit", None]:
            self.val_set = Subset(dataset=dataset, indices=val_idxs)
            self.train_set = Subset(dataset=dataset, indices=train_idxs)
        
        if stage in ["test", None]:
            self.test_set = Subset(dataset=dataset, indices=test_idxs)

    def compute_class_weights(self, dataset):
        dataset.targets = [dataset.dataset._labels[y] for y in dataset.indices]
        return super().compute_class_weights(dataset)
        
    def train_dataloader(self):
        return self.handle_imbalanced_dataset(
            dataset=self.train_set, 
            apply_class_weights=self.trainer.current_epoch >= self.hparams.oversample_epoch,
        )