"""Gunicorn configuration for the inventory parser."""

bind = "127.0.0.1:5001"
workers = 1
preload_app = False   # avoid PyTorch thread-pool fork deadlock
timeout = 300
