import csv
import os
import time

# TrainingLogger for statistics
class TrainingLogger:
    def __init__(self, log_file, event_names, action_names, log_frequency = 100):
        self.log_file = log_file
        self.event_names = event_names
        self.action_names = action_names
        self.log_frequency = log_frequency

        # Initialize statistics
        self.step_counter = 0
        self.round_counter = 0
        self.total_score = 0
        self.total_reward = 0
        self.cumulative_loss = 0
        self.total_steps = 0
        self.event_count = {event: 0 for event in self.event_names}
        self.action_count = {action: 0 for action in self.action_names}

        # Start time
        self.training_start_time = time.time()

        # Create or open CSV file to log statistics
        file_exists = os.path.isfile(self.log_file)
        with open(self.log_file, mode = 'a', newline = '') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write headers if file doesn't exist
                headers = ['Start Timestamp', 'Elapsed Time (s)', 'Rounds Played', 'Score', 'Total Reward', 'Loss', 'Steps'] + self.event_names + self.action_names
                writer.writerow(headers)

    def update_event_count(self, events):
        for event in events:
            if event in self.event_count:
                self.event_count[event] += 1

    def update_action_count(self, action):
        if action in self.action_count:
            self.action_count[action] += 1

    def update_score(self, score):
        self.total_score = score

    def update_reward(self, reward):
        self.total_reward += reward

    def update_loss(self, loss):
        self.cumulative_loss += loss

    def update_steps(self, steps):
        self.total_steps += steps

    def log_statistics(self):
        # Elapsed time
        elapsed_time = time.time() - self.training_start_time

        with open(self.log_file, mode = 'a', newline = '') as file:
            writer = csv.writer(file)
            row = [
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.training_start_time)),
                elapsed_time,
                self.round_counter,
                self.total_score,
                self.total_reward,
                self.cumulative_loss,
                self.total_steps
            ]

            #print(row)

            row += [self.event_count[event] for event in self.event_names]

            #print(row)

            row += [self.action_count[action] for action in self.action_names]

            #print(row)

            writer.writerow(row)

        # Reset cumulative statistics
        self.reset_statistics()

    def reset_statistics(self):
        self.round_counter = 0
        self.total_score = 0
        self.total_reward = 0
        self.cumulative_loss = 0
        self.total_steps = 0
        self.event_count = {event: 0 for event in self.event_names}
        self.action_count = {action: 0 for action in self.action_names}