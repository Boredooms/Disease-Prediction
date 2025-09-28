#!/usr/bin/env python3

import os
import multiprocessing

# Render.com configuration
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1  # Single worker for starter plan
threads = 2
max_requests = 1000
max_requests_jitter = 50
timeout = 300
keepalive = 5
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Memory management
worker_class = "sync"
worker_connections = 1000
max_worker_memory_usage = 400  # MB

# Process naming
proc_name = "mediai-disease-predictor"

# Security
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    worker.log.info("Worker aborted (pid: %s)", worker.pid)