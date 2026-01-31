"""
Utility functions for GPU management and system operations.
"""

import logging


def clear_gpu_memory():
    """Clear GPU memory."""
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass


def get_available_gpu_count():
    """Get number of available GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except:
        return 0


def print_gpu_allocation():
    """Print GPU memory allocation."""
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
            print(f"  Memory: {torch.cuda.mem_get_info(i)[0] / 1024**3:.1f}GB free / {torch.cuda.mem_get_info(i)[1] / 1024**3:.1f}GB total")
    except:
        print("No GPU information available")
