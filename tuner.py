from pathlib import Path
from typing import Any
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.models import resnet50
import numpy as np
import random
from multiprocessing import cpu_count
from tqdm import trange, tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.transforms import v2
import wandb

# To run: $ CUDA_VISIBLE_DEVICES=0 uv run tuner.py &

# -------------------------
# Config
# -------------------------
DEVICE = "cuda"
MODEL_NAME = "resnet50"
BS = 256
WD = 1e-3
EPOCHS = 100
WARMUP_EPOCHS = 5
NUM_WORKERS = cpu_count() // 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)


# Must be defined on the global scope to be picklable and accessible to workers
def set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32  # PyTorch auto increments its seed (internally) to get a unique seed per worker: "torch.initial_seed()" reflects that
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train_val(model, opt, epochs, train_loader, val_loader, run, scheduler=None):
    def closure(x, y):
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        return loss.item()

    best_val_acc = 0.0
    best_model: dict[str, Any] = {}
    best_epoch = 0

    for epoch in trange(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            n_batch = y.size(0)
            n_samples += n_batch
            loss = opt.step(lambda: closure(x, y))
            epoch_loss += loss * n_batch
            if scheduler:
                scheduler.step()

        val_acc = eval_model(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model.state_dict())
            best_epoch = epoch

        run.log(dict(
                train_loss=epoch_loss/n_samples,
                val_acc=val_acc
                ), step=epoch)

    run.summary["best_val_acc"] = best_val_acc
    run.summary["best_val_epoch"] = best_epoch
    return best_model


@torch.inference_mode()
def eval_model(model, eval_loader) -> float:
    model.eval()
    correct = 0
    total = 0

    for x, y in tqdm(eval_loader):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds.eq_(y)).sum_().item()
        total += y.size(0)

    acc = 100.0 * correct / total
    return acc


def main(data_dir: str):
    train_transform = v2.Compose(
        [
            v2.PILToTensor(),
            v2.RandomCrop(32, padding=4, padding_mode="reflect"),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=9),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]
    )

    eval_transform = v2.Compose(
        [
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    class TransformDataset(Dataset):
        def __init__(self, dataset, transforms):
            self.dataset = dataset
            self.T = transforms

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            x, y = self.dataset[idx]
            return self.T(x), y

    raw_ds = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
    )
    indices = list(range(len(raw_ds)))

    train_size = int(0.85 * len(raw_ds))  # 42,500

    test_ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BS, shuffle=False, num_workers=0, persistent_workers=False
    )

    # Start W&B Sweeps:
    run = wandb.init(
            project="bgd-tune-cifar10",
            config=dict(model=MODEL_NAME, epochs=EPOCHS, batch_size=BS, weight_decay=WD,)
            )

    config = run.config
    bgd_var = config.variant
    lr = config.lr
    seed = config.seed

    run.name = f"{bgd_var}_{lr}_{seed}_{MODEL_NAME}"
    run.group = f"{bgd_var}_{MODEL_NAME}"

    set_seed(seed)

    model = resnet50()
    model.fc = nn.Linear(2048, 10, bias=True)
    model.to(DEVICE)

    train_indices, val_indices = train_test_split(indices, train_size=train_size, stratify=raw_ds.targets, random_state=seed)

    train_ds, val_ds = Subset(raw_ds, train_indices), Subset(raw_ds, val_indices)
    train_ds, val_ds = TransformDataset(train_ds, train_transform), TransformDataset(val_ds, eval_transform)

    train_loader = DataLoader(train_ds,
                              batch_size=BS,
                              shuffle=True,
                              num_workers=NUM_WORKERS,  # torch pickles "init_fn" + dataset and all its transforms and sends serialized copy to each worker
                              persistent_workers=NUM_WORKERS > 0,
                              drop_last=False,
                              worker_init_fn=set_worker_seed,
                              generator=torch.Generator().manual_seed(seed))

    val_loader = DataLoader(val_ds, batch_size=500, shuffle=False, num_workers=0, persistent_workers=False)

    optimizer = bgd_var(model.parameters(), lr=lr, weight_decay=WD)

    # steps_per_epoch = len(train_loader)
    # total_steps = steps_per_epoch * EPOCHS
    # warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    # warmup_scheduler = LinearLR(
    #     optimizer,
    #     start_factor=0.01,
    #     end_factor=1.0,
    #     total_iters=warmup_steps
    # )
    #
    # cosine_scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=(total_steps - warmup_steps),
    #     eta_min=1e-4
    # )
    #
    # # Combine schedulers sequentially at the iteration level
    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, cosine_scheduler],
    #     milestones=[warmup_steps]
    # )

    best_model = train_val(model, optimizer, EPOCHS, train_loader, val_loader, run)

    ckpt_path = Path(run.dir) / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_model,
            "config": dict(config),
        },
        ckpt_path,
    )

    artifact = wandb.Artifact(
        name=f"{run.id}-best-model",
        type="model",
        metadata={
            "variant": bgd_var,
            "lr": lr,
            "seed": seed,
        },
    )

    artifact.add_file(str(ckpt_path))
    run.log_artifact(artifact, aliases=["best"])

    model.load_state_dict(best_model)
    test_acc = eval_model(model, test_loader)
    run.summary["test_acc"] = test_acc

    run.finish()
