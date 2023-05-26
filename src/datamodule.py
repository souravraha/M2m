from random import shuffle

import lightning as L
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as D
from torchvision import transforms as T

# Define the LightningDataModules
class SUN397DataModule(L.LightningDataModule):
    name = "sun397"
    NUM_CLASSES = 397
    MEANS = [0.4833, 0.4656, 0.4285]
    STDS = [0.2646, 0.2618, 0.2847]

    def __init__(self, data_dir, batch_size, val_split_per_class=10, test_split_per_class=40, **kwargs):
        super().__init__()
        self.data_dir = data_dir
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
        self.split = {
            "val": val_split_per_class,
            "test": test_split_per_class,
        }
        self.bs = batch_size

    def setup(self, stage=None):
        # Load and split the dataset into train and validation sets
        dataset = D.SUN397(root=self.data_dir, transform=self.transform)
        # Create a list of indices that belong to each class
        cls_idx_list = [[] for _ in range(len(dataset.classes))]
        for i, label in enumerate(dataset._labels):
            cls_idx_list[label].append(i)
        
        # Create val and test splits in a stratified manner
        val_idxs = []
        test_idxs = []
        for idx_list in cls_idx_list:
            shuffle(idx_list)
            val_idxs.extend(idx_list[:self.split["val"]])
            test_idxs.extend(idx_list[self.split["val"]:sum(self.split.values())])
        
        # The remaining are the train split
        train_idxs = list(set(range(len(dataset))) - set(val_idxs) - set(test_idxs))
        # Define subsets as required by the stage
        if stage == "fit":
            self.val_set = Subset(dataset=dataset, indices=val_idxs)
            self.train_set = Subset(dataset=dataset, indices=train_idxs)
        
        elif stage == "test":
            self.test_set = Subset(dataset=dataset, indices=test_idxs)
        
        else:
            raise NotImplementedError()
        
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, shuffle=True, num_workers=torch.get_num_threads(), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bs, num_workers=torch.get_num_threads(), pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, num_workers=torch.get_num_threads(), pin_memory=True)