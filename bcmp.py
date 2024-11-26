import simpy
import random
import numpy as np
import streamlit as st
from network_config import transition_matrix, node_config
from node import Node


def process(
    env: simpy.Environment,
    name: str,
    process_type: str,
    current_node: str,
    nodes: dict,
    transfer_delay_distribution: str = "normal",
    transfer_delay_params: dict = {"mean": 0.001, "std": 0.0001}
) -> None:
    """
    Simulates the process of a task moving through different nodes in the network.

    :param env: The simulation environment.
    :param name: The name of the process.
    :param process_type: The type of the process (either "user" or "system").
    :param current_node: The current node the process is at.
    :param nodes: A dictionary of nodes in the simulation network.
    :param transfer_delay_distribution: The distribution used for transfer delay ("normal", "exponential", "uniform").
    :param transfer_delay_params: Parameters for the transfer delay distribution.
    """

    while current_node != "End":
        # Process the task at the current node
        yield env.process(nodes[current_node].process(process_type))

        # Record entry time for "User"
        if current_node == "User":
            nodes[current_node].entry_times[name] = env.now

        # Calculate random transfer delay
        if transfer_delay_distribution == "normal":
            delay = max(0, np.random.normal(transfer_delay_params["mean"], transfer_delay_params["std"]))
        elif transfer_delay_distribution == "exponential":
            delay = np.random.exponential(transfer_delay_params["scale"])
        elif transfer_delay_distribution == "uniform":
            delay = np.random.uniform(transfer_delay_params["low"], transfer_delay_params["high"])
        else:
            raise ValueError("Unsupported distribution type for transfer delay.")

        # Simulate transfer delay
        yield env.timeout(delay)

        # Choose the next node based on the transition matrix
        next_node = random.choices(
            list(transition_matrix[current_node][process_type].keys()),
            weights=list(transition_matrix[current_node][process_type].values())
        )[0]
        current_node = next_node

    # Record exit time for "End"
    nodes["End"].exit_times[name] = env.now


def generate_processes(
    env: simpy.Environment,
    num_processes: int,
    external_arrival_rate: float,
    nodes: dict,
    transfer_delay_distribution: str = "normal",
    transfer_delay_params: dict = {"mean": 0.001, "std": 0.0001}
) -> None:
    """
    Generates new processes at random intervals based on a Poisson distribution.

    :param env: The simulation environment.
    :param num_processes: Total number of processes to simulate.
    :param external_arrival_rate: The rate (lambda) for the Poisson distribution.
    :param nodes: Dictionary of nodes in the network.
    """
    process_id = 1
    while process_id <= num_processes:
        process_type = random.choice(["user", "system"])  # Randomly choose process type
        yield env.timeout(np.random.poisson(1 / external_arrival_rate))  # Poisson inter-arrival times
        env.process(
            process(
                env,
                f"Process-{process_id}",
                process_type,
                "User",
                nodes,
                transfer_delay_distribution,
                transfer_delay_params
            )
        )
        process_id += 1


def calculate_total_arrival_rates(nodes: dict, external_arrival_rate: float) -> dict:
    """
    Calculate the total arrival rates to each node, including external and internal arrivals.

    :param nodes: Dictionary of Node objects.
    :param external_arrival_rate: External arrival rate to the network.
    :return: Dictionary of total arrival rates to each node.
    """
    total_arrival_rates = {name: 0 for name in nodes.keys()}
    total_arrival_rates["User"] = external_arrival_rate

    # Initialize a dictionary to store the arrival rates from other nodes
    internal_arrival_rates = {name: 0 for name in nodes.keys()}

    # Calculate the internal arrival rates
    for node_name, node in nodes.items():
        if node.queue_type == "FIFO":
            for next_node, prob in transition_matrix[node_name]["user"].items():
                internal_arrival_rates[next_node] += node.lambda_value * prob
            for next_node, prob in transition_matrix[node_name]["system"].items():
                internal_arrival_rates[next_node] += node.lambda_value * prob

    # Add the internal arrival rates to the total arrival rates
    for node_name in nodes.keys():
        total_arrival_rates[node_name] += internal_arrival_rates[node_name]

    return total_arrival_rates


def check_ergodicity(nodes: dict, external_arrival_rate: float) -> bool:
    """
    Checks if the ergodicity condition is satisfied for FIFO nodes in the network.
    If the ergodicity condition is satisfied for each FIFO node then the system is stable.

    :param nodes: Dictionary of Node objects type FIFO.
    :param external_arrival_rate: Overall arrival rate (lambda) for processes in the network.
    :return: None. Raises ValueError if any node violates the ergodicity condition.
    """
    total_arrival_rates = calculate_total_arrival_rates(nodes, external_arrival_rate)

    for name, node in nodes.items():
        if node.queue_type == "FIFO":
            # Calculate intensity (rho) for the FIFO node
            rho = total_arrival_rates[name] / (node.lambda_value * node.num_servers)

            if rho >= 1:
                st.error(
                    f"Ergodicity condition not satisfied for node '{name}': "
                    f"rho = {rho:.2f}. Adjust lambda_value or num_servers to ensure rho < 1."
                )
                return False
    return True


def run_simulation(
    sim_time: int,
    num_processes: int,
    external_arrival_rate: float,
    transfer_delay_distribution: str = "normal",
    transfer_delay_params: dict = {"mean": 0.001, "std": 0.0001},
) -> None:
    """
    Runs the network simulation, processes tasks, and generates statistics and plots.

    :param sim_time: The total simulation time.
    :param num_processes: The number of processes to simulate.
    :param external_arrival_rate: The rate at which new processes arrive in the system.
    :param transfer_delay_distribution: The distribution used for transfer delay ("normal", "exponential", "uniform").
    :param transfer_delay_params: Parameters for the transfer delay distribution.
    """
    # Initialize the simulation environment and nodes
    env = simpy.Environment()
    nodes = {name: Node(env, name, **node_config[name]) for name in node_config.keys()}

    # Start generating processes
    env.process(
        generate_processes(
            env,
            num_processes,
            external_arrival_rate,
            nodes,
            transfer_delay_distribution=transfer_delay_distribution,
            transfer_delay_params=transfer_delay_params
        )
    )

    # Run the simulation
    env.run(until=sim_time)

    # Collect statistics
    nodes['End'].processed_user = nodes['User'].processed_user
    nodes['End'].processed_system = nodes['User'].processed_system
    statistics = {
        name: (node.processed_user, node.processed_system) for name, node in nodes.items()
    }

    return {"nodes": nodes, "statistics": statistics}