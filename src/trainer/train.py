import torch.nn as nn
import torch


def train_step(model, target_model, buffer, device, gamma=0.99, batch_size=64):
    # Skip training this step if not enough samples in replay buffer
    if len(buffer) < batch_size:
        return

    # Create Adam optimizer for updating model's parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Mean Squared Error loss for comparing predicted Q-values to target Q-values
    loss_fn = nn.MSELoss()

    # Sample a batch of transitions from the replay buffer
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # Move all tensors to chosen device
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # Forward pass: compute Q-values for all actions in the current states
    q_values = model(states)

    # Compute Q-values for next states using the target network
    # Note that .max(dim=1)[0] extracts the maximum Q-value per row (per next state)
    next_q_values = target_model(next_states).max(dim=1)[0]

    # Compute the Bellman target:
    target = rewards + gamma * next_q_values * (1 - dones)

    # Select the Q-value corresponding to the action actually taken
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    # Compute loss between predicted Q-values and target Q-values
    loss = loss_fn(q_selected, target.detach())

    # Clear old gradients, backpropagate to compute new gradients, and update model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
