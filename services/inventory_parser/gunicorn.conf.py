"""Gunicorn configuration for the inventory parser."""

bind = "127.0.0.1:5001"
workers = 1
preload_app = True
timeout = 300


def post_fork(server, worker):
    """Re-initialise thread pools after fork so PyTorch/OpenBLAS use all cores."""
    import torch
    import os
    n = int(os.environ.get("OMP_NUM_THREADS", "4"))
    torch.set_num_threads(n)
