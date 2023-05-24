from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import datamodule, model

def train_tune(config):
    model = model.ERMModule(**config)
    datamodule = datamodule.SUN397DataModule()
    datamodule.setup()

    # Run your training and evaluation code here using model and datamodule
    # ...

    return metric_value  # Return the metric value to optimize (e.g., validation accuracy)

config = {
    'hidden_dim': tune.choice([32, 64, 128]),
    'learning_rate': tune.loguniform(1e-4, 1e-1)
}

scheduler = PopulationBasedTraining(
    time_attr='training_iteration',
    metric='val_loss',
    mode='min',
    perturbation_interval=3,
    hyperparam_mutations=config,
    resample_probability=0.25
)

analysis = tune.run(
    train_tune,
    config=config,
    num_samples=10,
    scheduler=scheduler
)

best_config = analysis.get_best_config(metric='val_loss')
print("Best config:", best_config)
