"""Gunicorn configuration for the Bond 100 Hall API.

Lightweight SQLite service — unlike the inventory_parser (PyTorch, single
worker, no preload) this can preload and run a few workers. SQLite WAL mode
(set in db.py) lets the workers read concurrently while one writes.
"""

bind = "127.0.0.1:5002"
workers = 2
preload_app = True
timeout = 30
