# network_config.py

# Transition matrix describing the system
transition_matrix = {
    "User": {
        "user": {"Process Scheduler": 1.0},
        "system": {"Process Scheduler": 1.0},
    },
    "Process Scheduler": {
        "user": {"IO Disk": 0.25, "IO Network": 0.15, "CPU": 0.6},
        "system": {"IO Disk": 0.4, "IO Network": 0.2, "CPU": 0.4},
    },
    "IO Disk": {
        "user": {"Memory": 0.6, "End": 0.4},
        "system": {"Memory": 0.6, "End": 0.4},
    },
    "IO Network": {
        "user": {"Memory": 0.6, "End": 0.4},
        "system": {"Memory": 0.6, "End": 0.4},
    },
    "CPU": {
        "user": {"Memory": 0.7, "System": 0.3},
        "system": {"Memory": 0.5, "System": 0.5},
    },
    "Memory": {
        "user": {"End": 1.0},
        "system": {"End": 1.0},
    },
    "System": {
        "user": {"Process Scheduler": 0.9, "End": 0.1},
        "system": {"Process Scheduler": 0.8, "End": 0.2},
    },
    "End": {
        "user": {},
        "system": {},
    },
}

# Node configuration with queue types and server counts
node_config = {
    "User": {"num_servers": 0, "queue_type": "IS", "lambda_value": 0.0},
    "Process Scheduler": {"num_servers": 2, "queue_type": "FIFO", "lambda_value": 11.0},
    "IO Disk": {"num_servers": 1, "queue_type": "FIFO", "lambda_value": 11.0},
    "IO Network": {"num_servers": 1, "queue_type": "FIFO", "lambda_value": 11.0},
    "CPU": {"num_servers": 2, "queue_type": "FIFO", "lambda_value": 11.0},
    "Memory": {"num_servers": 0, "queue_type": "IS", "lambda_value": 0.0},
    "System": {"num_servers": 1, "queue_type": "FIFO", "lambda_value": 11.0},
    "End": {"num_servers": 0, "queue_type": "IS", "lambda_value": 0.0},
}
