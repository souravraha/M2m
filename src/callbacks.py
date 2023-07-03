from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler


class M2mCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch >= pl_module.hparams.m2m_epoch:
            # Get the original training dataloader
            dataloader = trainer.train_dataloader

            if isinstance(dataloader.sampler, RandomSampler):
                pl_module.print("Sampler is a RandomSampler")
                # Create a new dataloader with class-balanced sampling
                dataset = dataloader.dataset
                sampler = WeightedRandomSampler(weights=[1 / trainer.datamodule.class_counts[y] for _, y in dataset], num_samples=len(dataset))
                balanced_dataloader = DataLoader(
                    dataset,
                    batch_size=dataloader.batch_size,
                    sampler=sampler,
                    num_workers=dataloader.num_workers,
                    pin_memory=dataloader.pin_memory,
                )

                # Override the trainer's train_dataloader with the balanced dataloader
                trainer.train_dataloader = balanced_dataloader

            elif isinstance(dataloader.sampler, WeightedRandomSampler):
                pl_module.print(f"Sampler is a WeightedRandomSampler in {pl_module.m2m_epoch}")