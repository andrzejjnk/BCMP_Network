import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import streamlit as st


def plot_queue_lengths_over_time(nodes: Dict[str, 'Node']) -> None:
    """
    Plots the queue lengths over time for FIFO nodes, 
    distinguishing between user and system processes.

    :param nodes: A dictionary of nodes, where the key is the node name and the value is the node object.
    """
    for name, node in nodes.items():
        if node.queue_type == "FIFO":
            # Prepare subplots for each node (2 columns: user/system)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"Queue Length Over Time - Node {name}", fontsize=16)

            # Plot for user processes
            axes[0].plot(node.time_log, node.queue_log_user, label="User Processes", color="#A50040")
            axes[0].set_xlabel("Simulation Time")
            axes[0].set_ylabel("Number of Processes in Queue")
            axes[0].set_title("User Processes")
            axes[0].legend()
            axes[0].grid()

            # Plot for system processes
            axes[1].plot(node.time_log, node.queue_log_system, label="System Processes", color="#005700")
            axes[1].set_xlabel("Simulation Time")
            axes[1].set_ylabel("Number of Processes in Queue")
            axes[1].set_title("System Processes")
            axes[1].legend()
            axes[1].grid()

            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout
            st.pyplot(fig)  # Render in Streamlit


def plot_average_waiting_times_per_node(nodes: Dict[str, 'Node']) -> None:
    """
    Plots average waiting times for FIFO nodes, distinguishing between user and system processes.
    
    :param nodes: A dictionary of nodes, where the key is the node name and the value is the node object.
    """
    # Only FIFO nodes for average waiting times
    fifo_nodes = [node for node in nodes.values() if node.queue_type == "FIFO"]
    fifo_node_names = [name for name, node in nodes.items() if node.queue_type == "FIFO"]
    
    # Calculate average waiting times for user and system processes
    avg_waiting_times_user = [
        (sum(node.waiting_times_user) / len(node.waiting_times_user) if node.waiting_times_user else 0)
        for node in fifo_nodes
    ]
    avg_waiting_times_system = [
        (sum(node.waiting_times_system) / len(node.waiting_times_system) if node.waiting_times_system else 0)
        for node in fifo_nodes
    ]

    # X positions for bars
    x = np.arange(len(fifo_node_names))
    bar_width = 0.4  # Width of the bars

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars for user processes
    user_bars = ax.bar(
        x - bar_width / 2,
        avg_waiting_times_user,
        width=bar_width,
        label="User Processes",
        color="#A50040"
    )

    # Bars for system processes
    system_bars = ax.bar(
        x + bar_width / 2,
        avg_waiting_times_system,
        width=bar_width,
        label="System Processes",
        color="#005700"
    )

    # Add labels above bars
    for bar in user_bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.001 if height > 0 else 0),  # Slightly above the bar
            f"{height:.2f}",
            ha='center',
            va='bottom',
            color="#A50040",
            fontsize=9
        )

    for bar in system_bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.001 if height > 0 else 0),  # Slightly above the bar
            f"{height:.2f}",
            ha='center',
            va='bottom',
            color="#005700",
            fontsize=9
        )

    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(fifo_node_names, rotation=45)

    # Labels and title
    ax.set_ylabel("Average Waiting Time")
    ax.set_title("Average Waiting Time in Queue (FIFO) - Process Types")

    # Add legend
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Render the plot in Streamlit
    st.pyplot(fig)


    # Waiting time histograms distinguishing process types
    for name, node in nodes.items():
        if node.queue_type == "FIFO":
            # Create figure and two subplots (1 row, 2 columns)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)  # sharey=True to share y-axis scale
            
            # User process histogram (left)
            if node.waiting_times_user:
                axes[0].hist(node.waiting_times_user, bins=20, color="#A50040", alpha=0.7, edgecolor="black", label="User Processes")
                axes[0].set_xlabel("Waiting Time")
                axes[0].set_ylabel("Number of Processes")
                axes[0].set_title(f"User: {name}")
                axes[0].legend()

            # System process histogram (right)
            if node.waiting_times_system:
                axes[1].hist(node.waiting_times_system, bins=20, color="#005700", alpha=0.7, edgecolor="black", label="System Processes")
                axes[1].set_xlabel("Waiting Time")
                axes[1].set_title(f"System: {name}")
                axes[1].legend()
            
            # Adjust layout and display
            fig.suptitle(f"Waiting Time Histograms: {name}", fontsize=14)  # Shared title for both plots
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for shared title
            st.pyplot(fig)



def plot_processed_processes_per_node(nodes: Dict[str, 'Node']) -> None:
    """
    Plots the number of processes processed by each node, distinguishing between user and system processes.
    
    :param nodes: A dictionary of nodes, where the key is the node name and the value is the node object.
    """
    # Node names
    node_names = list(nodes.keys())
    
    # Synchronize processed user and system processes in "End" with "User"
    nodes['End'].processed_user = nodes['User'].processed_user
    nodes['End'].processed_system = nodes['User'].processed_system
    
    # Processed user and system counts
    processed_user_counts = [node.processed_user for node in nodes.values()]
    processed_system_counts = [node.processed_system for node in nodes.values()]
    
    # X-axis for bars
    x = np.arange(len(node_names))  # X positions for nodes
    
    bar_width = 0.4  # Bar width

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars for user processes (shifted by -bar_width/2)
    user_bars = ax.bar(x - bar_width/2, processed_user_counts, width=bar_width, label="User Processes", color="#A50040")

    # Bars for system processes (shifted by +bar_width/2)
    system_bars = ax.bar(x + bar_width/2, processed_system_counts, width=bar_width, label="System Processes", color="#005700")

    # Add node names on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(node_names, rotation=45)

    # Labels and title
    ax.set_ylabel("Number of Processes")
    ax.set_title("Processed Processes in Each Node")

    # Show values on top of bars
    for bar in user_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{int(height)}", ha='center', va='bottom', color="#A50040", fontsize=9)

    for bar in system_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{int(height)}", ha='center', va='bottom', color="#005700", fontsize=9)

    # Legend and layout
    ax.legend()
    plt.tight_layout()

    # Render in Streamlit
    st.pyplot(fig)


def plot_time_spent_in_network_histogram(nodes: Dict[str, 'Node']) -> None:
    """
    Plots a histogram of the time spent in the network for processes.

    :param nodes: A dictionary of nodes, where the key is the node name and the value is the node object.
    """
    entry_times = nodes["User"].entry_times
    exit_times = nodes["End"].exit_times

    # Calculate times spent in the network
    network_times = [
        exit_times[process] - entry_times[process]
        for process in exit_times.keys()
        if process in entry_times
    ]

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(network_times, bins=20, color="#3498db", edgecolor="black", alpha=0.7)
    plt.title("Histogram of Times Spent in the Network")
    plt.xlabel("Time in Network")
    plt.ylabel("Number of Processes")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Render histogram in Streamlit
    st.pyplot(plt.gcf())
