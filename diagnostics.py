import os
import sys
import torch

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available.")
    # Check if we are using the CPU-only version of torch
    if "+cu" not in torch.__version__ and "cuda" not in torch.__version__:
        print("Suggestion: You seem to be using the CPU-only version of PyTorch.")
        print("Try installing the CUDA version with:")
        print("pip install torch --index-url https://download.pytorch.org/whl/cu118")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import failed: {e}")
