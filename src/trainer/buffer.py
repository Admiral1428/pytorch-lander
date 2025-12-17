import game.constants as cfg
from collections import deque
import random
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # Store a single transition tuple in the buffer, representing one step of experience
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Retrieves a random minibatch of stored experience tuples and returns
    # them as PyTorch tensors so the agent can learn from past transitions
    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    # Return the current number of stored transitions.
    def __len__(self):
        return len(self.buffer)
