import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Locks the global RNG across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f" Global RNG seeded to {seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False