import torch.nn as nn


# Reinforcement learning model
class LanderNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Define a simple feed‑forward neural network:
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),  # First fully connected layer
            nn.ReLU(),  # Nonlinear activation
            nn.Linear(128, 128),  # Second fully connected layer
            nn.ReLU(),  # Nonlinear activation
            nn.Linear(128, action_dim),  # Output layer: one Q-value per action
        )

    # Forward pass: simply run input x through the defined network
    def forward(self, x):
        return self.net(x)
