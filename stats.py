from typing import Dict

# Function to generate network statistics
def calculate_network_stats(nodes: Dict[str, 'Node'], total_simulation_time: float) -> None:
    """
    Calculates and prints network statistics for each node in the simulation.
    
    This includes:
    - Average queue length for user and system processes, based on Little's Law.
    - Average waiting time for user and system processes.
    - Arrival rate of user and system processes.
    
    :param nodes: A dictionary containing the nodes in the network, where the key is the node name and the value is the node object.
    :param total_simulation_time: The total simulation time for the network.
    """
    print("Network Statistics:\n")
    for name, node in nodes.items():
        if node.queue_type == "FIFO":
            print(f"Node {name}:")

            # Calculate average waiting time
            avg_waiting_time_user = sum(node.waiting_times_user) / len(node.waiting_times_user) if node.waiting_times_user else 0
            avg_waiting_time_system = sum(node.waiting_times_system) / len(node.waiting_times_system) if node.waiting_times_system else 0

            # Calculate arrival rate (number of processes / total simulation time)
            arrival_rate_user = node.processed_user / total_simulation_time if total_simulation_time > 0 else 0
            arrival_rate_system = node.processed_system / total_simulation_time if total_simulation_time > 0 else 0

            # Average queue length using Little's Law
            avg_queue_length_user = arrival_rate_user * avg_waiting_time_user if avg_waiting_time_user > 0 else 0
            avg_queue_length_system = arrival_rate_system * avg_waiting_time_system if avg_waiting_time_system > 0 else 0

            print(f"  Average Queue Length (User): {avg_queue_length_user:.2f}")
            print(f"  Average Queue Length (System): {avg_queue_length_system:.2f}\n")
