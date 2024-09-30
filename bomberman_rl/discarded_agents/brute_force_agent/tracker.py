import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import pandas as pd

Q_TABLE_PATH = os.path.join(os.path.dirname(__file__), "q_table.pkl")
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def print_q():
    with open(Q_TABLE_PATH, "rb") as file:
        q_table = pickle.load(file)

    for state, actions in list(q_table.items())[:10]:
        print(f"State: {state}, Q-values: {actions}")

def visualize_q():
    with open(Q_TABLE_PATH, "rb") as file:
        q_table = pickle.load(file)
    
    states = list(q_table.keys())
    up_values = [q_table[state][0] for state in states]
    right_values = [q_table[state][1] for state in states]
    down_values = [q_table[state][2] for state in states]
    left_values = [q_table[state][3] for state in states]
    wait_values = [q_table[state][4] for state in states]
    bomb_values = [q_table[state][5] for state in states]

    plt.plot(up_values, label = "UP")
    plt.plot(right_values, label = "RIGHT")
    plt.plot(down_values, label = "DOWN")
    plt.plot(left_values, label = "LEFT")
    plt.plot(wait_values, label = "WAIT")
    plt.plot(bomb_values, label = "BOMB")

    plt.legend()
    plt.show()

def visualize_training_data(file_path = "../../training_data.txt"):
    # Read data
    data = pd.read_csv(file_path, sep="\t")
    # Sum iterations
    data['Cumulative_iterations'] = data['Iterations'].cumsum()
    
    # Plot training time
    plt.figure(figsize = (12, 7))

    plt.subplot(2, 2, 1)
    plt.plot(data['Cumulative_iterations'], data['Training_time'], marker = 'o')
    plt.title('Training time over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Training time [s]')

    # Plot total reward over iterations
    plt.subplot(2, 2, 2)
    plt.plot(data['Cumulative_iterations'], data['Total_reward'], marker = 'o')
    plt.title('Reward over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Total Reward')

    # Plot action frequency
    plt.subplot(2, 2, 3)
    for action in ACTIONS:
        plt.plot(data['Cumulative_iterations'], data[action], marker = 'o', label = action)
        plt.title('Action Count Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Action Count')
    plt.legend()      

    plt.tight_layout()
    plt.show  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-table Tracker")
    parser.add_argument("--print", action="store_true", help="Print the first 10 Q-values")
    parser.add_argument("--visualize", action="store_true", help="Visualize the Q-values for UP and RIGHT actions")
    parser.add_argument("--visualize-training", action="store_true", help="Visualize the training data")

    args = parser.parse_args()

    if args.print:
        print_q()

    if args.visualize:
        visualize_q()

    if args.visualize_training:
        visualize_training_data()
