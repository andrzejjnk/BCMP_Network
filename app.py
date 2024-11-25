import streamlit as st
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
        yield env.process(nodes[current_node].process(process_type))
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
        yield env.timeout(np.random.poisson(1 / arrival_rate))  # Poisson inter-arrival times
        env.process(process(env, f"Process-{process_id}", process_type, "User", nodes))
        process_id += 1


def run_simulation(sim_time: int, num_processes: int, arrival_rate: float) -> dict:
    """
    Runs the network simulation and returns the results.

    :param sim_time: Total simulation time.
    :param num_processes: Number of processes to simulate.
    :param arrival_rate: Rate (lambda) for the Poisson distribution of process arrivals.
    :return: A dictionary with simulation results, including nodes and statistics.
    """
    # Initialize the simulation environment and nodes
    env = simpy.Environment()
    nodes = {name: Node(env, name, **node_config[name]) for name in node_config.keys()}

    # Start generating processes
    env.process(generate_processes(env, num_processes, arrival_rate, nodes))

    # Run the simulation
    env.run(until=sim_time)

    # Collect statistics
    nodes['End'].processed_user = nodes['User'].processed_user
    nodes['End'].processed_system = nodes['User'].processed_system
    statistics = {
        name: (node.processed_user, node.processed_system) for name, node in nodes.items()
    }

    return {"nodes": nodes, "statistics": statistics}


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("BCMP Network Simulation")
    st.sidebar.header("Simulation Parameters")

    # Sidebar inputs
    sim_time = st.sidebar.slider("Simulation Time", min_value=100, max_value=5000, step=50, value=1000)
    num_processes = st.sidebar.slider("Number of Processes", min_value=50, max_value=5000, step=50, value=1000)
    arrival_rate = st.sidebar.slider("Arrival Rate (Processes/Unit Time)", min_value=0.1, max_value=50.0, step=0.1, value=10.0)

    # Run simulation button
    if st.sidebar.button("Run Simulation"):
        st.write("### Simulation Results")
        st.write(f"Simulation Time: {sim_time}, Number of Processes: {num_processes}, Arrival Rate: {arrival_rate}")

        # Run the simulation
        results = run_simulation(sim_time, num_processes, arrival_rate)
        nodes = results["nodes"]
        statistics = results["statistics"]

        # Display statistics
        st.write("#### Processed Processes per Node:")
        stats_table = []
        for node, (user, system) in statistics.items():
            stats_table.append({"Node": node, "User Processes": user, "System Processes": system})
        st.table(stats_table)

        # Generate plots
        st.write("#### Queue Lengths Over Time:")
        plot_queue_lengths_over_time(nodes)

        st.write("#### Average Waiting Times Per Node:")
        plot_average_waiting_times_per_node(nodes)

        st.write("#### Processed Processes Per Node:")
        plot_processed_processes_per_node(nodes)


if __name__ == "__main__":
    main()
