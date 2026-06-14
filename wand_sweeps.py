import wandb
from bgd import BGD_VARIANTS
from tuner import main, PROJECT_NAME

variant_names: list[str] = [bgd_var.__name__ for bgd_var in BGD_VARIANTS]
SEEDS = (77, 433, 1024)
LRs = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5)

# 1. Define the sweep configuration
sweep_configuration = {
    "name": "bgd-resnet50-cifar10",
    "method": "grid",  # 'grid' tries every combination. Use 'bayes' or 'random' for large searches.
    "metric": {
        "name": "test_acc",
        "goal": "maximize",
    },
    "parameters": {
        "variant": {"values": variant_names},
        "lr": {"values": LRs},
        "seed": {"values": SEEDS},
    },
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    # 2. Initialize the sweep on W&B servers
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=PROJECT_NAME,
    )
    print(f"Sweep ID: {sweep_id}")

    wandb.agent(
        sweep_id=sweep_id,
        function=lambda: main(data_dir=args.data_dir),
        project=PROJECT_NAME,
    )
