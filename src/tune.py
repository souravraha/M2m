import os
import datamodule
import module
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import LightningConfigBuilder, LightningTrainer
from ray.tune.schedulers.pb2 import PB2
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--num_epochs", help="Enter your name", required=True)
# parser.add_argument("--num_samples", help="Enter your age", required=True)
# parser.add_argument("--city", help="Enter your city", required=True)
# args = parser.parse_args()

# The maximum training epochs
num_epochs = 5

# Number of sampls from parameter space
num_samples = 10

accelerator = "gpu"

config = {
    "layer_1_size": tune.choice([32, 64, 128]),
    "layer_2_size": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-1),
}

if SMOKE_TEST:
    num_epochs = 3
    num_samples = 3
    accelerator = "cpu"

dm = datamodule.SUN397DataModule(
    data_dir="./data", batch_size=128
)
logger = TensorBoardLogger(save_dir=os.getcwd(), name="tune-ptl-example", version=".")

lightning_config = (
    LightningConfigBuilder()
    .module(
        cls=module.ERMModule, 
        backbone_name="ResNet", 
        backbone_hparams={
            "num_classes": datamodule.SUN397DataModule.NUM_CLASSES,
            "num_blocks": [2, 2, 2, 2],
            "c_hidden": [64, 128, 256, 512],
            "act_fn_name": "relu",
            "block_name": "preact",
        },
        optimizer_name="SGD",
        optimizer_hparams={
            "lr": 0.1, 
            # "momentum": 0.9, "weight_decay": 2e-4
        },
    )
    .trainer(max_epochs=num_epochs, accelerator=accelerator, logger=logger)
    .fit_params(datamodule=dm)
    .checkpointing(monitor="val_MulticlassRecall", save_top_k=2, mode="max")
    .build()
)

# Make sure to also define an AIR CheckpointConfig here
# to properly save checkpoints in AIR format.
run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="val_MulticlassRecall",
        checkpoint_score_order="max",
    ),
)

# scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

scaling_config = ScalingConfig(
    num_workers=3, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
)

if SMOKE_TEST:
    scaling_config = ScalingConfig(
        num_workers=3, use_gpu=False, resources_per_worker={"CPU": 1}
    )

# Define a base LightningTrainer without hyper-parameters for Tuner
lightning_trainer = LightningTrainer(
    scaling_config=scaling_config,
    run_config=run_config,
)

# def tune_mnist_asha(num_samples=10):
#     scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

#     tuner = tune.Tuner(
#         lightning_trainer,
#         param_space={"lightning_config": lightning_config},
#         tune_config=tune.TuneConfig(
#             metric="val_MulticlassRecall",
#             mode="max",
#             num_samples=num_samples,
#             scheduler=scheduler,
#         ),
#         run_config=air.RunConfig(
#             name="tune_mnist_asha",
#         ),
#     )
#     results = tuner.fit()
#     best_result = results.get_best_result(metric="val_MulticlassRecall", mode="max")
#     return best_result


# tune_mnist_asha(num_samples=num_samples)


def tune_mnist_pbt(num_samples=10):
    # The range of hyperparameter perturbation.
    mutations_config = (
        LightningConfigBuilder()
        .module(
            config={
                "lr": tune.loguniform(1e-4, 1e-1),
            }
        )
        .build()
    )

    

    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": lightning_config},
        tune_config=tune.TuneConfig(
            metric="val_MulticlassRecall",
            mode="max",
            num_samples=num_samples,
            # Create a PBT scheduler
            scheduler=PB2(
                perturbation_interval=1,
                time_attr="training_iteration",
                hyperparam_mutations={"lightning_config": mutations_config},
            ),
        ),
        run_config=air.RunConfig(
            name="tune_mnist_pbt",
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="val_MulticlassRecall", mode="max")
    return best_result

tune_mnist_pbt(num_samples=num_samples)
