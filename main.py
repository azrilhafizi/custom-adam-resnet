import wandb
import yaml
from train import train

# Load sweep configuration from YAML file
with open("sweep_config.yaml") as file:
    sweep_config = yaml.safe_load(file)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="custom_adam_resnet")

# Start the sweep
wandb.agent(sweep_id, function=train, count=10)