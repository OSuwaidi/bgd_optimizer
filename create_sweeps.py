import wandb
from bgd import BGD_VARIANTS

# To run: $uv run create_sweep.py --> prints <entity/project/sweep_id>

variant_names: list[str] = [bgd_var.__name__ for bgd_var in BGD_VARIANTS]
PROJECT_NAME = "bgd-tune-cifar10"
SEEDS = (77, 433, 1024)
LRs = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

if __name__ == "__main__":
    # 1. Define the sweep configuration
    sweep_configuration = {
        "program": "tuner.py",
        "name": "momentum-bgd-cifar10",
        "method": "grid",  # 'grid' tries every combination. Use 'bayes' or 'random' for large searches.
        "metric": {
            "name": "test_acc",
            "goal": "maximize",
            },
        "parameters": {
            "variant": {"values": variant_names},
            "ema": {"values": (True, False)},
            "absorb": {"values": (True, False)},
            "lr": {"values": LRs},
            "seed": {"values": SEEDS},
            },
        }

    # 2. Initialize the sweep on W&B servers
    sweep_id = wandb.sweep(
            sweep=sweep_configuration,
            project=PROJECT_NAME,
            )
    print(f"Sweep ID: {sweep_id}")

    # wandb.agent(
    #         sweep_id=sweep_id,
    #         function=lambda: main(**args_dict),
    #         project=PROJECT_NAME,
    #         )
