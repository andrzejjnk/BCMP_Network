import streamlit as st

from network_config import node_config
from node import Node
from plots import plot_average_waiting_times_per_node, plot_processed_processes_per_node, plot_queue_lengths_over_time, plot_time_spent_in_network_histogram
from bcmp import run_simulation, check_ergodicity


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("BCMP Network Simulation")
    bcmp_model_png = "BCMP_Network_Model.png"
    
    # Wyświetlenie obrazu z podpisem
    st.image(bcmp_model_png, caption="BCMP Network Model", use_column_width=True)
    st.sidebar.header("Simulation Parameters")

    # Initialize session state for ergodicity and change tracking
    if "ergodicity_valid" not in st.session_state:
        st.session_state.ergodicity_valid = False
    if "changes_made" not in st.session_state:
        st.session_state.changes_made = False

    # Sidebar inputs for simulation parameters
    sim_time = st.sidebar.number_input("Simulation Time", min_value=100, max_value=5000, step=50, value=1000)
    num_processes = st.sidebar.number_input("Number of Processes", min_value=50, max_value=5000, step=50, value=1000)
    arrival_rate = st.sidebar.number_input("Arrival Rate (Processes entering to the system/Unit Time)", min_value=0.01, max_value=100.0, step=0.01, value=10.0, on_change=lambda: st.session_state.update({"changes_made": True})
)

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
                min_value=0.001,
                max_value=100.0,
                step=0.1,
                format="%.3f",
                value=node_config[node_name]["lambda_value"],
                on_change=lambda: st.session_state.update({"changes_made": True})
            )
            node_lambdas[node_name] = lambda_value

    # Update node_config with user-defined lambdas
    for node_name in node_config.keys():
        if node_config[node_name]["queue_type"] == "FIFO":
            node_config[node_name]["lambda_value"] = node_lambdas[node_name]

    # Sidebar for number of servers configuration
    st.sidebar.subheader("Node Server Count Configuration")
    node_servers = {}
    for node_name in node_config.keys():
        if node_config[node_name]["queue_type"] == "FIFO":
            num_servers = st.sidebar.number_input(
                f"Number of Servers ({node_name})",
                min_value=1,
                max_value=100,
                step=1,
                value=node_config[node_name].get("num_servers", 1),  # Default to 1 server if not defined
                on_change=lambda: st.session_state.update({"changes_made": True})
            )
            node_servers[node_name] = num_servers

    # Update node_config with user-defined number of servers
    for node_name in node_config.keys():
        if node_config[node_name]["queue_type"] == "FIFO":
            node_config[node_name]["num_servers"] = node_servers[node_name]

    # Check ergodicity button
    if st.sidebar.button("Check Ergodicity"):
        try:
            ergodicity_valid = check_ergodicity(
                {name: Node(None, name, **node_config[name]) for name in node_config.keys()},
                arrival_rate
            )
            st.session_state.ergodicity_valid = ergodicity_valid
            st.session_state.changes_made = False  # Reset changes_made flag
            if ergodicity_valid:
                st.success("All nodes satisfy the ergodicity condition!")
            else:
                st.error("Ergodicity condition not satisfied for some nodes.")
        except ValueError as e:
            st.session_state.ergodicity_valid = False
            st.error(str(e))

    # Display a message if changes were made and ergodicity needs to be checked
    if st.session_state.changes_made:
        st.session_state.ergodicity_valid = False
        st.warning("Changes were made. Please click 'Check Ergodicity' to validate the new configuration.")

    # Run simulation button (enabled only if ergodicity condition is satisfied)
    if st.sidebar.button("Run Simulation", disabled=not st.session_state.ergodicity_valid):
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
