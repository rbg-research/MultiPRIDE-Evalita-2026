import os
import random
import numpy as np
import torch

def set_all_seeds(seed=42):
    """
    Set random seeds for Python, NumPy, PyTorch (CPU and GPU) for reproducibility.
    
    Args:
        seed: Integer seed value (e.g., 42)
    """
    # Python's built-in random module
    random.seed(seed)
    
    # Environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU (all devices)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Disable PyTorch's cuDNN auto-tuning (ensures deterministic behavior)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Can be slower but more reproducible
    
    # Optional: Use deterministic algorithms (PyTorch 1.9+)
    torch.use_deterministic_algorithms(True)
    
    print(f"✓ All random seeds set to {seed}")