# network_config.py

# Transition matrix describing the system
transition_matrix = {
    "User": {
        "user": {"CPU Scheduler": 1.0},
        "system": {"CPU Scheduler": 1.0},
    },
    "CPU Scheduler": {
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
        "user": {"CPU Scheduler": 0.9, "End": 0.1},
        "system": {"CPU Scheduler": 0.8, "End": 0.2},
    },
    "End": {
        "user": {},
        "system": {},
    },
}

# Node configuration with queue types and server counts
node_config = {
    "User": {"num_servers": 0, "queue_type": "IS"},
    "CPU Scheduler": {"num_servers": 2, "queue_type": "FIFO"},
    "IO Disk": {"num_servers": 1, "queue_type": "FIFO"},
    "IO Network": {"num_servers": 1, "queue_type": "FIFO"},
    "CPU": {"num_servers": 2, "queue_type": "FIFO"},
    "Memory": {"num_servers": 0, "queue_type": "IS"},
    "System": {"num_servers": 1, "queue_type": "FIFO"},
    "End": {"num_servers": 0, "queue_type": "IS"},
}