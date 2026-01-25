import random
import torch


# Returns the action, and whether it was random
def select_action(model, state, action_dim_choice, epsilon):
    with torch.no_grad():
        q_values = model(state.unsqueeze(0))
        max_q = q_values.max().item()
        mean_q = q_values.mean().item()

    # With probability epsilon, choose a random action (exploration)
    if random.random() < epsilon:
        # 0 = no thrust or torque
        # 1 = thrust
        # 2 = left torque
        # 3 = right torque
        # 4 = thrust + left torque
        # 5 = thrust + right torque
        return random.randint(0, action_dim_choice - 1), True, max_q, mean_q
    # Otherwise, choose the best action according to the Q-network (exploitation)
    else:
        return torch.argmax(q_values).item(), False, max_q, mean_q
