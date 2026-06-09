import torch
import numpy as np
import random

def set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def flatten_image(x):
    return x.view(-1)
