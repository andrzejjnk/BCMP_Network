# node.py

import simpy
import random

class Node:
    """
    Class representing a node in the network.
    Each node can either process or pass through processes depending on the configuration.
    """
    def __init__(self, env, name, num_servers, queue_type, lambda_value):
        """
        Initializes the Node with the given parameters.
        
        :param env: Simulation environment
        :param name: Name of the node
        :param num_servers: Number of servers available for the node
        :param queue_type: Type of queue used ('FIFO' or 'IS')
        """
        self.env = env
        self.name = name
        self.lambda_value = lambda_value
        self.num_servers = num_servers
        self.queue_type = queue_type
        self.queue = simpy.Resource(env, capacity=num_servers) if queue_type == "FIFO" else None
        self.processed_user = 0
        self.processed_system = 0
        self.user_times = []  # User process service times
        self.system_times = []  # System process service times
        self.waiting_times_user = []  # User waiting times (FIFO)
        self.waiting_times_system = []  # System waiting times (FIFO)
        
        # Queue length logs
        self.queue_log_user = []  # Queue length for user processes over time
        self.queue_log_system = []  # Queue length for system processes over time
        self.time_log = []  # Time stamps

        self.entry_times = {}  # Dictionary to store entry times of processes
        self.exit_times = {}   # Dictionary to store exit times of processes


    def log_queue_length(self):
        """
        Logs the current queue length for user and system processes.
        """
        if self.queue_type == "FIFO":
            user_queue_length = len([req for req in self.queue.queue if hasattr(req, 'process_type') and req.process_type == "user"])
            system_queue_length = len([req for req in self.queue.queue if hasattr(req, 'process_type') and req.process_type == "system"])
            self.queue_log_user.append(user_queue_length)
            self.queue_log_system.append(system_queue_length)
            self.time_log.append(self.env.now)


    def process(self, process_type):
        """
        Processes a user or system type process at the node.

        :param process_type: Type of the process ('user' or 'system')
        """
        start_time = self.env.now

        if self.queue_type == "FIFO":
            request_time = self.env.now  # Time when the process enters the queue
            with self.queue.request() as req:
                req.process_type = process_type
                self.log_queue_length()  # Log queue length
                yield req  # Wait for resource
                waiting_time = self.env.now - request_time  # Waiting time in queue
                if process_type == "user":
                    self.waiting_times_user.append(waiting_time)
                elif process_type == "system":
                    self.waiting_times_system.append(waiting_time)

                # Service time using exponential distribution
                yield self.env.timeout(random.expovariate(self.lambda_value))
                self.log_queue_length()  # Update queue length after service

        elif self.queue_type == "IS":
            yield self.env.timeout(0)  # No waiting time, immediate service

        end_time = self.env.now
        time_spent = end_time - start_time

        if process_type == "user":
            self.processed_user += 1
            self.user_times.append(time_spent)
        elif process_type == "system":
            self.processed_system += 1
            self.system_times.append(time_spent)
