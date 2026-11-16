"""Seeds, devices and small helpers."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch (CPU and every GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    """The given device, else the environment variable FP_DEVICE, else CUDA if available, else the CPU."""
    name = name or os.environ.get("FP_DEVICE")
    if name is None:
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def sync(device: torch.device) -> None:
    """Waits for the GPU (for wall-clock timings)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
