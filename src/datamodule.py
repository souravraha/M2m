# Define the LightningDataModule

from functools import partial
from random import shuffle

import lightning as L
import torch
import torch.utils.data as data
from torchvision import datasets as D
from torchvision import transforms as T
from tqdm import tqdm


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
        self.dataloader = partial(data.DataLoader, 
            dataset=dataset, batch_size=self.bs, 
            num_workers=torch.get_num_threads(), 
            pin_memory=True,
        )
        self.cls_idxs = [[] for _ in range(SUN397DataModule.NUM_CLASSES)]
        for i, label in tqdm(enumerate(dataset._labels)):
            self.cls_idxs[label].append(i)

        self.val_idxs = []
        self.test_idxs = []
        
        for idx_list in tqdm(self.cls_idxs):
            shuffle(idx_list)
            self.val_idxs.extend(idx_list[:self.split["val"]])
            self.test_idxs.extend(idx_list[self.split["val"]:sum(self.split.values())])
        
        self.train_idxs = list(set(range(len(dataset))) - set(self.val_idxs) - set(self.test_idxs))

    def train_dataloader(self):
        return self.dataloader(sampler=data.SubsetRandomSampler(self.train_idxs))

    def val_dataloader(self):
        return self.dataloader(sampler=data.SubsetRandomSampler(self.val_idxs))

    def test_dataloader(self):
        return self.dataloader(sampler=data.SubsetRandomSampler(self.test_idxs))