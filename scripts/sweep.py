import wandb

wandb.login()


def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    wandb.init(project="novae_swav")
    score = objective(wandb.config)
    wandb.log({"score": score})


# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "train/loss_epoch"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="novae_swav")

    wandb.agent(sweep_id, function=main, count=10)
