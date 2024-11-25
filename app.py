import streamlit as st
import simpy
import random
import numpy as np

from network_config import transition_matrix, node_config
from node import Node
from plots import plot_average_waiting_times_per_node, plot_processed_processes_per_node, plot_queue_lengths_over_time, plot_time_spent_in_network_histogram


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
    entry_time = env.now if current_node == "User" else None

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
    arrival_rate: float,
    nodes: dict,
    transfer_delay_distribution: str = "normal",
    transfer_delay_params: dict = {"mean": 0.001, "std": 0.0001}
) -> None:
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


def run_simulation(
    sim_time: int,
    num_processes: int,
    arrival_rate: float,
    transfer_delay_distribution: str = "normal",
    transfer_delay_params: dict = {"mean": 0.001, "std": 0.0001},
) -> None:
    """
    Runs the network simulation, processes tasks, and generates statistics and plots.

    :param sim_time: The total simulation time.
    :param num_processes: The number of processes to simulate.
    :param arrival_rate: The rate at which new processes arrive in the system.
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
            arrival_rate,
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


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("BCMP Network Simulation")
    bcmp_model_png = "BCMP_Network_Model.png"
    
    # Wyświetlenie obrazu z podpisem
    st.image(bcmp_model_png, caption="BCMP Network Model", use_column_width=True)
    st.sidebar.header("Simulation Parameters")

    # Sidebar inputs for simulation parameters
    sim_time = st.sidebar.number_input("Simulation Time", min_value=100, max_value=5000, step=50, value=1000)
    num_processes = st.sidebar.number_input("Number of Processes", min_value=50, max_value=5000, step=50, value=1000)
    arrival_rate = st.sidebar.number_input("Arrival Rate (Processes/Unit Time)", min_value=0.01, max_value=50.0, step=0.01, value=10.0)
    #lambda_value = st.sidebar.number_input("Service time (lambda parameter for exponential distribution)", min_value=0.01, max_value=10.0, step=0.1, value=1.0)

    # Sidebar inputs for transfer delay distribution
    st.sidebar.header("Transfer Delay Parameters")
    transfer_delay_distribution = st.sidebar.selectbox(
        "Transfer Delay Distribution",
        options=["normal", "exponential", "uniform"],
        index=0
    )

    # Set parameters based on the chosen distribution
    if transfer_delay_distribution == "normal":
        mean = st.sidebar.number_input("Mean (Normal)", 
                                        min_value=0.001, 
                                        max_value=1.0, 
                                        value=0.001, 
                                        step=0.001, 
                                        format="%.3f")
        std = st.sidebar.number_input("Standard Deviation (Normal)", 
                                    min_value=0.000010, 
                                    max_value=1.0, 
                                    value=0.000010, 
                                    step=0.000010, 
                                    format="%.6f")
        transfer_delay_params = {"mean": mean, "std": std}

    elif transfer_delay_distribution == "exponential":
        scale = st.sidebar.number_input("Scale (Exponential)", 
                                        min_value=0.001, 
                                        max_value=1.0, 
                                        value=0.001, 
                                        step=0.000010, 
                                        format="%.6f")
        transfer_delay_params = {"scale": scale}

    elif transfer_delay_distribution == "uniform":
        low = st.sidebar.number_input("Low (Uniform)", 
                                    min_value=0.001, 
                                    max_value=1.0, 
                                    value=0.001, 
                                    step=0.000001, 
                                    format="%.6f")
        high = st.sidebar.number_input("High (Uniform)", 
                                        min_value=0.001, 
                                        max_value=1.0, 
                                        value=0.001, 
                                        step=0.000001, 
                                        format="%.6f")
        transfer_delay_params = {"low": low, "high": high}

    # Sidebar for lambda configuration
    st.sidebar.subheader("Node Service Time Configuration")
    node_lambdas = {}
    for node_name in node_config.keys():
        if node_config[node_name]["queue_type"] == "FIFO":
            lambda_value = st.sidebar.number_input(
                f"Service Time Lambda ({node_name})",
                min_value=0.1,
                max_value=100.0,
                step=0.1,
                value=node_config[node_name]["lambda_value"]
            )
            node_lambdas[node_name] = lambda_value

    # Update node_config with user-defined lambdas
    for node_name in node_config.keys():
        if node_config[node_name]["queue_type"] == "FIFO":
            node_config[node_name]["lambda_value"] = node_lambdas[node_name]


    # Run simulation button
    if st.sidebar.button("Run Simulation"):
        st.write("### Simulation Results")
        st.write(f"Simulation Time: {sim_time}, Number of Processes: {num_processes}, Arrival Rate: {arrival_rate}")
        st.write(f"Transfer Delay Distribution: {transfer_delay_distribution}")
        st.write(f"Transfer Delay Parameters: {transfer_delay_params}")

        # Run the simulation
        results = run_simulation(
            sim_time=sim_time,
            num_processes=num_processes,
            arrival_rate=arrival_rate,
            transfer_delay_distribution=transfer_delay_distribution,
            transfer_delay_params=transfer_delay_params,
        )
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

        # Histogram of times spent in the network
        st.write("Histogram of Times Spent in the Network")
        plot_time_spent_in_network_histogram(nodes)

        # Calculate and display network statistics
        st.write("### Detailed Network Statistics")
        with st.expander("View Network Statistics"):
            stats_output = []
            for name, node in nodes.items():
                if node.queue_type == "FIFO":
                    # Calculate average waiting time
                    avg_waiting_time_user = sum(node.waiting_times_user) / len(node.waiting_times_user) if node.waiting_times_user else 0
                    avg_waiting_time_system = sum(node.waiting_times_system) / len(node.waiting_times_system) if node.waiting_times_system else 0

                    # Calculate arrival rate (number of processes / total simulation time
                    arrival_rate_user = node.processed_user / sim_time if sim_time > 0 else 0
                    arrival_rate_system = node.processed_system / sim_time if sim_time > 0 else 0

                    # Average queue length using Little's Law
                    avg_queue_length_user = arrival_rate_user * avg_waiting_time_user if avg_waiting_time_user > 0 else 0
                    avg_queue_length_system = arrival_rate_system * avg_waiting_time_system if avg_waiting_time_system > 0 else 0

                    stats_output.append(f"Node {name}:\n"
                                        f"  Average Queue Length (User): {avg_queue_length_user:.2f}\n"
                                        f"  Average Queue Length (System): {avg_queue_length_system:.2f}\n"
                                        f"  Arrival Rate (User): {arrival_rate_user:.2f}\n"
                                        f"  Arrival Rate (System): {arrival_rate_system:.2f}\n"
                                        f"  Average Waiting Time (User): {avg_waiting_time_user:.2f}\n"
                                        f"  Average Waiting Time (System): {avg_waiting_time_system:.2f}\n")
            
            for stats in stats_output:
                st.text(stats)



if __name__ == "__main__":
    main()
