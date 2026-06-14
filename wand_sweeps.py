import wandb

from bgd import (
    CGRS,
    DGRS,
    DGRF,
    CGRF,
    DGSS,
    CGSS,
    DGSF,
    CGSF,
    DPRS,
    CPRS,
    DPRF,
    CPRF,
    DPSS,
    CPSS,
    DPSF,
    CPSF,
)

BGD_VARIANTS = (
    DGRS,
    CGRS,
    DGRF,
    CGRF,
    DGSS,
    CGSS,
    DGSF,
    CGSF,
    DPRS,
    CPRS,
    DPRF,
    CPRF,
    DPSS,
    CPSS,
    DPSF,
    CPSF,
)

variant_names = [bgd_var.__name__ for bgd_var in BGD_VARIANTS]
SEEDS = (77, 433, 1024)
LRs = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5)

# 1. Define the sweep configuration
sweep_configuration = {
    "method": "grid",  # 'grid' tries every combination. Use 'bayes' or 'random' for large searches.
    "parameters": {
        "variant": {"values": variant_names},
        "lr": {"values": LRs},
        "seed": {"values": SEEDS},
    },
}

# 2. Initialize the sweep on W&B servers
# This returns a unique sweep_id
sweep_id = wandb.sweep(sweep=sweep_configuration, project="bgd-tune-cifar10")
