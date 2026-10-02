"""Gunicorn configuration for the anonymous feedback API."""

bind = "127.0.0.1:5003"
workers = 1
preload_app = True
timeout = 30
accesslog = "-"
