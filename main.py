import simpy
import random
import numpy as np

from network_config import transition_matrix, node_config
from node import Node
from plots import plot_average_waiting_times_per_node, plot_processed_processes_per_node, plot_queue_lengths_over_time
from stats import calculate_network_stats


def process(env: simpy.Environment, name: str, process_type: str, current_node: str, nodes: dict) -> None:
    """
    Simulates the process of a task moving through different nodes in the network.

    :param env: The simulation environment.
    :param name: The name of the process.
    :param process_type: The type of the process (either "user" or "system").
    :param current_node: The current node the process is at.
    :param nodes: A dictionary of nodes in the simulation network.
    """
    while current_node != "End":
        # Process the task at the current node
        yield env.process(nodes[current_node].process(process_type))
        
        # Choose the next node based on the transition matrix
        next_node = random.choices(
            list(transition_matrix[current_node][process_type].keys()),
            weights=list(transition_matrix[current_node][process_type].values())
        )[0]
        current_node = next_node


def generate_processes(env: simpy.Environment, num_processes: int, arrival_rate: float, nodes: dict) -> None:
    """
    Generates new processes at random intervals based on a Poisson distribution.
    
    :param env: The simulation environment.
    :param num_processes: Total number of processes to simulate.
    :param arrival_rate: The rate (lambda) for the Poisson distribution.
    :param nodes: Dictionary of nodes in the network.
    """
    process_id = 1
    while process_id <= num_processes:
        process_type = random.choice(["user", "system"])  # Randomly choose process type
        yield env.timeout(np.random.poisson(1/arrival_rate))  # Poisson inter-arrival times
        env.process(process(env, f"Process-{process_id}", process_type, "User", nodes))  # Start the process
        process_id += 1

# Zrobic sredni czas przejscia od węzła początkowego do końca dla każdego procesu, dorobić jakieś ywkresy, histogramy
def run_simulation() -> None:
    """
    Runs the network simulation, processes tasks, and generates statistics and plots.
    This function encapsulates the simulation logic.

    The simulation:
    1. Initializes the nodes and environment.
    2. Creates processes for user and system tasks.
    3. Runs the simulation for the specified time.
    4. Outputs statistics and generates plots.

    :return: None
    """
    SIM_TIME = 1000  # Simulation time
    NUM_PROCESSES = 1000  # Number of processes to simulate
    ARRIVAL_RATE = 10  # Average number of processes arriving per unit time (Poisson distribution)

    # Initialize the simulation environment and nodes
    env = simpy.Environment()
    nodes = {name: Node(env, name, **node_config[name]) for name in node_config.keys()}

    # Start generating processes based on Poisson distribution
    env.process(generate_processes(env, NUM_PROCESSES, ARRIVAL_RATE, nodes))

    # Run the simulation until the specified time
    env.run(until=SIM_TIME)

    # Output the statistics
    print("Statistics:")
    nodes['End'].processed_user = nodes['User'].processed_user
    nodes['End'].processed_system = nodes['User'].processed_system
    for name, node in nodes.items():
        print(f"{name}: {node.processed_user} user processes, {node.processed_system} system processes")

    # Perform additional calculations (e.g., network stats)
    calculate_network_stats(nodes, SIM_TIME)

    # Generate plots
    plot_queue_lengths_over_time(nodes)
    plot_average_waiting_times_per_node(nodes)
    plot_processed_processes_per_node(nodes)


def main() -> None:
    """
    Entry point of the program. This function calls the `run_simulation` function
    to execute the simulation.
    """
    run_simulation()


if __name__ == "__main__":
    main()
