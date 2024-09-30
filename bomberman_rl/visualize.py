import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse


### Load data
def load_data(agent_name, file_path = None):
    """
    Load csv file into a pd DataFrame
    """

    # Default file_path if not otherwise specified
    if file_path is None:
        file_path = f"agent_code/{agent_name}/training_stats.csv"

    if not os.path.isfile(file_path):
        print(f"File not found in: {file_path}")
        sys.exit(1)

    data = pd.read_csv(file_path)
    return data


### Event statistics
def plot_event_statistics(data, stats_to_plot):
    """
    Plot selected statistics over cumulative number of training iterations

    :param data: DataFrame w data
    :param stats_to_plot: List of columns to plot
    """

    plt.figure(figsize = (14, 7))

    # Calculate cumulative training iterations
    data["Cumulative Iterations"] = data["Rounds Played"].cumsum()

    for stat in stats_to_plot:
        if stat in data.columns:
            plt.plot(data["Cumulative Iterations"], data[stat] / data["Rounds Played"], 
                     label = stat, linewidth = 0, marker = 'o')
        else:
            print(f"Statistic {stat} not found in columns.")

    plt.title("Statistics")
    plt.xlabel("Cumulative training iterations")
    plt.ylabel("Normalized Value per Round")
    plt.legend()
    plt.tight_layout()

    plt.show()

### Comparison event staistics
def compare_event_statistics(data1, data2, stats_to_plot, label1, label2):
    """
    Plot and compare the statistics over cumulative number of training iterations 
    for two different agents.
    """
    
    plt.figure(figsize = (14, 7))

    # Calculate cumulative training iterations
    data1["Cumulative Iterations"] = data1["Rounds Played"].cumsum()
    data2["Cumulative Iterations"] = data2["Rounds Played"].cumsum()

    for stat in stats_to_plot:
        if stat in data1.columns:
            plt.plot(data1["Cumulative Iterations"], data1[stat] / data1["Rounds Played"], 
                label = f"{label1} - {stat}")
        else: 
            print(f"Statistic {stat} not found in {label1} columns.")

        if stat in data2.columns:
            plt.plot(data2["Cumulative Iterations"], data2[stat] / data2["Rounds Played"], 
                label = f"{label2} - {stat}")
        else: 
            print(f"Statistic {stat} not found in {label2} columns.")

    plt.title("Statistics Comparison")
    plt.xlabel("Cumulative training iterations")
    plt.ylabel("Normalized Value per Round")
    plt.legend()
    plt.tight_layout()

    plt.show()


### Histogram
def plot_action_histogram(data, actions_to_plot, row = -1):
    """
    Plot a histogram of the action distribution of selected row
    """
    if row >= len(data) or row < -len(data):
        print(f"Invalid row index: {row}. Data only has {len(data)} rows")
        sys.exit(1)

    row_data = data.iloc[row]

    action_counts = []
    for action in actions_to_plot:
        if action in data.columns:
            action_counts.append(row_data[action])
        else:
            print(f"Action {action} not found in columns.")

    plt.figure(figsize = (10, 6))
    plt.bar(actions_to_plot, action_counts, color = 'skyblue')
    plt.title(f"Action distribution for Row {row if row >= 0 else 'Last' }")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.show()

### Comparison Histogram
def compare_action_histogram(data1, data2, actions_to_plot, row, label1, label2):
    """
    Compare action distributions between two agents for the given row
    """
    if row >= len(data1) or row < -len(data1):
        print(f"Invalid row index: {row}. Data only has {len(data1)} rows")
        sys.exit(1)

    if row >= len(data2) or row < -len(data2):
        print(f"Invalid row index: {row}. Data only has {len(data2)} rows")
        sys.exit(1)

    row_data1 = data1.iloc[row]
    row_data2 = data2.iloc[row]

    action_counts1 = []
    action_counts2 = []
    
    for action in actions_to_plot:
        if action in data1.columns:
            action_counts1.append(row_data1[action])
        else:
            print(f"Action {action} not found in {label1} columns.")
        
        if action in data2.columns:
            action_counts2.append(row_data2[action])
        else:
            print(f"Action {action} not found in {label2} columns.")

    # Plot histograms side by side for comparison
    plt.figure(figsize=(10, 6))
    
    width = 0.35  # Width of bars
    x = range(len(actions_to_plot))

    plt.bar([i - width/2 for i in x], action_counts1, width=width, color='skyblue', label=label1)
    plt.bar([i + width/2 for i in x], action_counts2, width=width, color='orange', label=label2)

    plt.title(f"Action distribution comparison for Row {row if row >= 0 else 'Last'}")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.xticks(x, actions_to_plot)
    plt.legend()
    plt.tight_layout()

    plt.show()

def main():
    parser = argparse.ArgumentParser(description = "Visualize training statistics.")

    # Basic mode for a single agent
    parser.add_argument('--agent1', type = str, required = True, help = "Name of the (first) agent under which to search for training_stats.csv file")

    # File paths for statistics
    parser.add_argument('--file-path', type = str, required = False, help = "Search for stats in desired directory.")
    parser.add_argument('--file-path2', type = str, required = False, help = "Search for stats in desired directory for second agent.")

    # Compare to second agent
    parser.add_argument('--agent2', type = str, required = False, help = "Name of the second agent under which to search for training_stats.csv file")

    # Time series
    parser.add_argument('--stats', nargs = '+', help = "Select statistics to visualize over training iterations")

    # Time series comparison
    parser.add_argument('--stats-comparison', nargs = '+', help = "Select statistics to visualize and compare between agents")

    # Histogram
    parser.add_argument('--hist', nargs = '*', help = "Select action counts to visualize in histogram")
    parser.add_argument('--row', type = int, default = -1, help = "Row index for histogram. Default: Last row")

    args = parser.parse_args()

    # If only one agent, plot single-agent stats
    if args.agent1 and not args.agent2:
        data = load_data(args.agent1, file_path = args.file_path)

        # Stats plot?
        if args.stats:
            plot_event_statistics(data, args.stats)

        # Hist?
        if args.hist is not None:
            default_actions = ["MOVED_UP", "MOVED_DOWN", "MOVED_RIGHT", "MOVED_LEFT", "BOMB_DROPPED", "WAITED", "INVALID_ACTION"]
            actions_to_plot = args.hist if args.hist else default_actions
            plot_action_histogram(data, actions_to_plot, row = args.row)

    elif args.agent1 and args.agent2:
        data1 = load_data(args.agent1, file_path = args.file_path)
        data2 = load_data(args.agent2, file_path = args.file_path2)

        # Stats comparison plot?
        if args.stats:
            compare_event_statistics(data1, data2, args.stats, label1 = args.agent1, label2 = args.agent2)

        # Comparison hist?
        if args.hist is not None:
            default_actions = ["MOVED_UP", "MOVED_DOWN", "MOVED_RIGHT", "MOVED_LEFT", "BOMB_DROPPED", "WAITED", "INVALID_ACTION"]
            actions_to_plot = args.hist if args.hist else default_actions
            compare_action_histogram(data1, data2, actions_to_plot, row = args.row, label1 = args.agent1, label2 = args.agent2)

    else:
        print("Please provide at least one agent for visualization.")


if __name__ == "__main__":
    main()