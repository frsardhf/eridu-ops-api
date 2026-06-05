"""Gunicorn configuration for the Bond 100 Hall API.

Lightweight SQLite service — unlike the inventory_parser (PyTorch, single
worker, no preload) this can preload and run a few workers. SQLite WAL mode
(set in db.py) lets the workers read concurrently while one writes.
"""

bind = "127.0.0.1:5002"
workers = 2
preload_app = True
timeout = 30

# Access log to stdout → journald. nginx already logs request lines, but this
# confirms what actually reached the app (e.g. a submission that got past the
# edge rate limit) alongside the app's own submission-outcome logs.
accesslog = "-"
