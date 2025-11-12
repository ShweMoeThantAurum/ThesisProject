"""
Sets deterministic random seeds for reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_global_seed(seed=42):
    """Sets deterministic seeds for NumPy, random, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[Reproducibility] Global seed fixed at {seed}")
