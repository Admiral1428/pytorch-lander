# pytorch-lander
*PyTorch Lander* is a 2D Python game that simulates rocket landings, blending classic arcade gameplay with modern AI control. Inspired by [*Lunar Lander (1979)*](https://www.arcade-museum.com/Videogame/lunar-lander), the project challenges players to guide a descending rocket safely onto a landing pad across procedurally generated terrain. The simulation supports autonomous control via PyTorch, showcasing reinforcement learning and AI-driven decision-making.

<img width="3188" height="1794" alt="12-09-25" src="https://github.com/user-attachments/assets/88c34510-4f13-412f-a6a5-ebd71d431677" />

https://github.com/user-attachments/assets/6813e678-4339-40f4-8592-38676eb4cb30

## Key features:
- Procedural terrain generation with reproducible random seeds
- Physics-based rocket model with 3 degrees of freedom (Z-rotation, X-Y translation)
- Robust collision detection for terrain impact and successful landing validation
- Dual control modes: player inputs or PyTorch-trained AI agents

The purpose of this project was to practice programming skills in Python and PyTorch by blending game development, physics simulation, and AI/ML integration. The intent was to create a playable experience and a technical showcase.

## Installation

PyTorch’s runtime libraries are extremely large, and packaging them into a standalone ``.exe`` produces a multi‑gigabyte file. For practicality, the project is provided as source code instead of a compiled executable. Please follow these steps to install the project:
  * Download and install Python 3.13.5 or newer.
  * Add your Python installation folder and the associated ``Scripts`` folder to your system environment variables "Path" variable.
  * Download the project source code and unzip, or clone the repository using Git.
  * Install the project's dependencies from the root directory by running the following command: ``pip install -r requirements.txt``
  * You can then run the game or training program directly from the source code. 

## Design and Modeling

Similar to the arcade game *Lunar Lander (1979)*, the goal was to create a playable game where the player or AI agent could apply thrust or torque to steer the rocket towards a horizontal landing pad, amidst vertically varied terrain representing mountains and valleys. The `PyGame` library was utilized to draw the level, player, and text to a window that can be resized by the user. To streamline the game code, key classes were created representing the `Rocket`, `Level`, and `Game` logic. 

The `Rocket` class includes various physical properties such as mass, size, and inertia. Various methods are included to update the rocket state, including physics calculations of position, velocity, and acceleration. Simple kinematic relations were utilized to compute the trajectory over the course of each time step. This begins with calculating the mass and inertia based on any applied thrust or torque: 

$$m_{fuel} = m_{fuel} - \dot{m}_{thrust}*dt - \dot{m}_{torque}*dt$$
$$m = m_{empty} + m_{fuel}$$
$$I_z = {{1}\over{4}} m \left({{h}\over{2}}\right)^2 + {{1}\over{12}} m w^2$$

The sum of forces and moments is used to determine translational and rotational acceleration.

$$\Sigma F_x = F \text{cos}(\theta) = ma_x$$
$$\Sigma F_y = F \text{sin}(\theta) - mg = ma_y$$
$$\Sigma T_z = \tau = I_z\alpha $$

<img width="2090" height="1123" alt="image" src="https://github.com/user-attachments/assets/65bdbd5c-189e-4be3-9dd4-b1597811405f" />

Since the acceleration terms are constant at a given time step, the integration of velocity is approximated over the small step as:

$$v_x = v_{xi} + a_x dt$$
$$v_y = v_{yi} + a_y dt$$
$$\omega = \omega_{i} + \alpha dt$$

A similar approach is used to integrate position:

$$x = x_i + v_x dt$$
$$y = y_i + v_y dt$$
$$\theta = \theta_i + \alpha dt$$

These states are updated at every time step, and are used to calculate other attributes such as the outer boundary position of the rocket. A simple grid of $(x_{boundary},y_{boundary})$ points is defined for the outside of the rocket, and is translated and rotated as follows:

$$\begin{bmatrix}
x_{boundary}' \\
y_{boundary}' \\
\end{bmatrix} = 
\begin{bmatrix}
\text{cos}(\theta) & \text{sin}(\theta) \\
-\text{sin}(\theta) & \text{cos}(\theta) \\
\end{bmatrix}
\begin{bmatrix}
x_{boundary} \\
y_{boundary} \\
\end{bmatrix} + 
\begin{bmatrix}
x \\
y \\
\end{bmatrix}$$

The `Level` class initializes the position and appearance of the terrain, landing pad, sky, and rocket starting location. The `Game` class handles the initialization of `PyGame`, sound effects and images, screen updates, and detection of terminal events. Several external modules were also defined with input parsing functions and constant parameters.

## Training - Setup

A single trajectory results in one of these possible terminal events:
* Collision with terrain
* Successful landing
* Rocket escape off-screen

At a given time step during the episode, the rocket is capable of the following actions during human control:
* No thrust or torque
* Thrust
* Left torque
* Right torque
* Thrust and left torque
* Thrust and right torque

For simplicity, the latter two actions were excluded from the AI agent to reduce the action dimension. Therefore, the agent could pick one of four possible actions at a given time step.

The `PyTorch` library was used to initialize and train the AI agent in different phases of increasing complexity, using Reinforcement Learning (RL). To expedite training, the simulation rate was limited to 10 Hz. When the model is re-inserted into the main game loop that runs at 60 Hz, the actions are limited to 10 Hz to remain consistent with the training. During training, the physics and state calculations are computed faster than real-time by using a fixed value for the time step, rather than a value representing elapsed time between displayed frames. A status message is shown in the terminal for the outcome of each episode, rather than displaying the game window with each trajectory:

<img width="2979" height="350" alt="image" src="https://github.com/user-attachments/assets/e87f6507-396d-4088-b407-da017beee1af" />
<br><br>

After initializing the `Game`, `Level`, and `Rocket` classes, the device for training calculations was set to the GPU if available (training utilized an NVIDIA RTX 3080 with [CUDA 12.6](https://download.pytorch.org/whl/cu126)):

`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

Actual calculations of rocket trajectory, collision logic, etc. would still be handled by the CPU, but all tensor operations for training were expedited by using the GPU. Several classes were initialized for handling training:
* `ReplayBuffer` - Stores past RL transitions and returns random minibatches of them as PyTorch tensors so the agent can learn from previous experience
* `LanderNet` - Defines a feed-forward neural network that takes a state vector and outputs one Q-value per possible action for the RL agent

At each training step, a state vector is computed which is converted into tensor form. It consists of the following data:
* Normalized rocket position (X and Y)
* Normalized rocket velocity (X and Y)
* Sine and cosine of the rocket angle
* Normalized rocket angular velocity
* Normalized rocket fuel quantity
* Normalized rocket position relative to landing pad (X and Y)
* Terrain height values within a limited horizontal range of the rocket

Once the state is computed and an action is selected, various shaping rewards are calculated depending on the training phase. These are described in more detail in subsequent sections.

Terminal rewards are also included for collision, pad contact, escape, over-rotation, or landing. These only occur at the last time step of the episode, whereas the shaping rewards occur for every time step.

## Training Phase 1 - Descent and Horizontal Movement

The first training phase consisted of rewarding horizontal and vertical progression towards the landing pad. The following hyperparameters were utilized:

| Hyperparameter | Description |
|----------------|-------------|
| **Epsilon start:** `ε = 1.0` | Agent begins training with full exploration, choosing random actions 100% of the time. |
| **Epsilon end:** `ε = 0.2` | Exploration decays to a minimum of 20% to prevent overfitting and maintain some randomness. |
| **Epsilon decay:** `4500` episodes | Controls how quickly ε moves from 1.0 to 0.2, shifting behavior from exploration to exploitation. |
| **Discount factor:** `γ = 0.99` | Makes the agent value long‑term cumulative reward almost as much as immediate reward. |
| **Max episodes:** `5000` | Caps episode length to prevent infinite or unproductive trajectories. |
| **Replay buffer size:** `100,000` | Maximum number of past transitions stored for sampling during training. |
| **Update interval:** `4` steps | Number of environment steps between each training update, determining how often the network learns. |
| **Warmup steps:** `2000` steps | Number of preliminary steps before learning is allowed to begin. |
| **Evaluation interval:** `200` episodes | How often to conduct assessment of model with full exploitation. |

The shaping and terminal rewards were limited to the following:

| Type | Name | Reward or Penalty? |
|------|------|--------------------|
| Shaping | Angle | Penalty |
| Shaping | Vertical Position Relative to Pad | Penalty |
| Shaping | Horizontal Position Relative to Pad | Penalty |
| Shaping | Terrain Proximity | Penalty |
| Shaping | Vertical Velocity | Penalty / Reward |
| Shaping | Time | Penalty |
| Shaping | Fuel | Penalty |
| Terminal | Escape | Penalty = -600 |
| Terminal | Over-Rotation beyond 90 degrees | Penalty = -400 |
| Terminal | Crashed into Terrain | Penalty = -175 |
| Terminal | Pad contact | Reward = 200 |
| Terminal | Landing | Reward = 200 |

When the model was evaluated every 200 episodes with $\varepsilon = 0$, the training plot and model were exported if the pad contact rate exceeded 25 percent. It was determined that 100 percent pad contact occurred at episode 3800 of 5000, which is also near the peak pad contact rate during training. Evaluation with full exploitation was necessary, since relying on the training contact rate alone was not necessarily an indication of training success.

Another way to expedite training was to introduce a success buffer, which is populated with steps from the successful pad contact or landing episodes. This buffer is used more frequently than the main buffer, since the latter contains failure trajectories with less useful behaviors after exploration has revealed successful cases. The following policy was utilized:
| Rolling Pad Contact Rate | Frequency of Success Buffer Usage |
|------|------|
| < 20% | 40% |
| < 50% | 50% |
| < 75% | 60% |
| ≥ 75% | 70% |

A series of plots were created to help diagnose behaviors for tuning the shaping terms and hyperparameters, and a subset is shown here which illustrates high pad contact rates near episode 3800:

### **High pad contact rate at episode 3800:**
<img width="3004" height="1176" alt="lander_model_phase_01_ 02_event_rates  - arrow" src="https://github.com/user-attachments/assets/b9561fd9-b907-4855-b76c-4508c5c77bd8" />

### **Vertical distance to pad is near zero at episode 3800:**
<img width="3038" height="1176" alt="lander_model_phase_01_ 04_vert_dist  - arrow" src="https://github.com/user-attachments/assets/73621118-f31d-4560-bb31-0262e3f1929c" />

### **Horizontal distance to pad is near zero at episode 3800:**
<img width="3038" height="1176" alt="lander_model_phase_01_ 05_horz_dist  - arrow" src="https://github.com/user-attachments/assets/b2ad6263-2a6b-4d86-9c38-12444805b662" />

### **Trajectory visualization with green success and blue failure:**
<img width="1398" height="1361" alt="training_trajectories_episode_3800_rate_100" src="https://github.com/user-attachments/assets/29496ae6-9298-457f-9149-21178705219c" />

Given this convergence, the model was saved into an external file `lander_model_phase_01.pth` and could be used within the main game loop to produce descent:

https://github.com/user-attachments/assets/e188e5a0-bdc1-49fd-95ae-fa7f03c7dccd

## Future Training Phases

Future training will focus on achieving a fully successful landing, defined by the following criteria:
* Horizontal velocity less than safe threshold
* Vertical velocity less than safe threshold
* Landing angle within tolerance (nearly upright)
* Entirety of rocket positioned on pad horizontally
* Rocket is touching pad vertically within tolerance

Achieving this will likely require an additional round of Phase 1 training with a higher initial starting altitude to give the agent more time to stabilize its descent from normal starting altitudes. Subsequently, new shaping terms can be introduced for the final landing phase—rewarding safe velocities, upright angles, and precise pad alignment. Each stage would initialize from the previous model checkpoint to accelerate convergence and preserve learned behaviors.

Training so far has shown that the agent is extremely sensitive to hyperparameters, shaping magnitudes, and terminal reward structure. This sensitivity has been one of the most interesting aspects of the project: the agent frequently discovers locally optimal but undesirable behaviors, and careful reward design is required to guide it toward true success cases.
