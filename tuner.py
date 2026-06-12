import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.models import resnet50
import numpy as np
import random
from multiprocessing import cpu_count
from tqdm import trange
import torch.nn.functional as F
from torchvision.transforms import v2

# -------------------------
# Config
# -------------------------
DEVICE = "cuda"
SEEDS = (77, 433, 1024)
BS = 256
EPOCHS = 90
NUM_WORKERS = cpu_count() // 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)


def set_worker_seed(worker_id):  # must be defined on the global scope to be picklable and accessed by workers
    worker_seed = torch.initial_seed() % 2 ** 32  # PyTorch auto increments its seed to get a unique seed per worker ("torch.initial_seed()" reflects that)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train(model, opt, epochs, train_loader):
    for _ in trange(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.inference_mode()
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(DEVICE, ), y.to(DEVICE, )
        logits = model(x)

        preds = logits.argmax(dim=1)
        correct += (preds.eq(y)).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc * 100:.2f}%")


transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.2860,), (0.3530,)),
    ])

total_ds = datasets.CIFAR10(root="~/Python/datasets/CIFAR10", train=True, download=True, transform=transform)
train_size = int(0.8 * len(total_ds))  # 40,000
val_size = len(total_ds) - train_size  # 10,000

test_ds = datasets.CIFAR10(root="~/Python/datasets/CIFAR10", train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=0, persistent_workers=False)

model = resnet50()
model.fc = nn.Linear(2048, 10, bias=True)
model.to(DEVICE)


train_ds, val_ds = random_split(
    total_ds,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_ds,
                          batch_size=BS,
                          shuffle=True,
                          num_workers=NUM_WORKERS,  # torch pickles "init_fn" + dataset and all its transforms and sends serialized copy to each worker
                          persistent_workers=NUM_WORKERS > 0,
                          drop_last=True,
                          worker_init_fn=set_worker_seed,
                          generator=torch.Generator().manual_seed(SEED))
vs_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0, persistent_workers=True)

train(model, sgd, EPOCHS, train_loader)

for handle in handles:
    handle.remove()

test(model, test_loader)
