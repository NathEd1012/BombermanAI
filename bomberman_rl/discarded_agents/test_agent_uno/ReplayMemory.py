from collections import deque
import random
from collections import namedtuple

"""
Replay Memory: To store the Transitions our agent observes and reuse this data as part of the 
training process.

This step is important to keep the Replay Memory of experiences (state, action, reward, next state)
buffered and then be able to sample from this memory buffer to train the agent. According to ChatGPT,
this helps "break the correlation between consecutive experiences and stabilizes the learning process"

This is because if your agent trains from consecutive experiences, these experiences are
correlated. By sampling from a memory of random transitions, you can decorrelate the learning
for better results.

The tools we require for this are:

    deque: double-ended queue to append and pop items efficiently, used to store transitions with
    a fixed maximum size REPLAY_MEMORY_SIZE. When the memory is full, old experiences are 
    replaced with newer ones.

    namedtuple: apparently the same as a tuple but with named fields for readability. This makes
    the Transition = ('state', 'action', ...) clearer later on.
"""

# Transition
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        # Save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Sample a random batch of transitions from the memory buffer
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # Current length of the memory
        return len(self.memory)