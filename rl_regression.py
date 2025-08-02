# -*- coding: utf-8 -*-
"""RL regression.ipynb
# yongchao.huang@abdn.ac.uk

# Stage 1: a working but not good example.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 1. Setup, Seeds, and Hyperparameters ---
# Set a seed for reproducibility
SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For full reproducibility on CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
EPOCHS = 500
BATCH_SIZE = 64
N_SAMPLES = 1000
SIGMA = 0.2  # Sigma for the Gaussian reward kernel
NOISE_STD = 0.1 # Exploration noise

# --- 2. Create the Synthetic Dataset ---
# The agent needs to learn the function y = sin(x)
# With the seed set, this data will be the same every time
x_data = np.random.uniform(-np.pi, np.pi, N_SAMPLES)
y_data = np.sin(x_data) + np.random.normal(0, 0.1, N_SAMPLES) # Add some noise

# Convert to PyTorch tensors
X = torch.from_numpy(x_data).float().unsqueeze(1).to(device)
Y = torch.from_numpy(y_data).float().unsqueeze(1).to(device)
dataset = torch.utils.data.TensorDataset(X, Y)

# Create a generator for the DataLoader to ensure reproducible shuffling
g = torch.Generator()
g.manual_seed(SEED)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)


# --- 3. Define Actor and Critic Networks ---
# The network weights will now be initialized identically on each run

# The Actor network (the policy) maps state -> action
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() # Tanh activation to bound the output between -1 and 1, similar to sin(x)
        )

    def forward(self, state):
        return self.net(state)

# The Critic network (the Q-function) maps (state, action) -> value
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), # Input is state and action concatenated
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        # Concatenate state and action along the feature dimension
        state_action = torch.cat([state, action], 1)
        return self.net(state_action)

# --- 4. Training ---
actor = Actor().to(device)
critic = Critic().to(device)

# Define optimizer and loss function (criterion) separately
actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)
critic_criterion = nn.MSELoss() # MSELoss is in torch.nn

actor_losses = []
critic_losses = []

print("Starting training...")
for epoch in range(EPOCHS):
    for states, true_ys in dataloader:
        states, true_ys = states.to(device), true_ys.to(device)

        # --- Update Critic ---
        # Get actions from the actor and add exploration noise
        actions = actor(states)
        # The noise will be the same on each run due to the torch seed
        noise = (torch.randn_like(actions) * NOISE_STD).to(device)
        noisy_actions = torch.clamp(actions + noise, -1, 1)

        # Calculate the real reward using the Gaussian kernel
        # We detach this so its gradient isn't computed during the critic's update
        rewards = torch.exp(-(true_ys - noisy_actions).pow(2) / (2 * SIGMA**2)).detach()

        # Critic predicts the value of the (state, noisy_action) pair
        q_values = critic(states, noisy_actions)

        # Critic loss is the difference between predicted value and actual reward
        # Use the predefined criterion
        critic_loss = critic_criterion(q_values, rewards)

        # Update critic network
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # --- Update Actor ---
        # Actor's goal is to output actions that the critic gives a high value to.
        # We want to maximize the critic's output, so we minimize its negative.
        # We use the original, non-noisy actions for the policy update.
        actor_loss = -critic(states, actor(states)).mean()

        # Update actor network
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

    actor_losses.append(actor_loss.item())
    critic_losses.append(critic_loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

print("Training finished.")


# --- 5. Visualization ---
# Generate a clean set of x values for plotting the learned function
# the range to match the training data distribution
x_test = torch.linspace(-10, 10, 300).unsqueeze(1).to(device)

# Get the actor's predictions (actions) for the test data
with torch.no_grad():
    y_pred = actor(x_test).cpu().numpy()

# Create a plot showing final errors across the state space
final_errors = []
with torch.no_grad():
    all_preds = actor(x_test)
    final_errors = (np.sin(x_test) - all_preds).abs().cpu().numpy().flatten()

plt.figure(figsize=(18, 5))

# Plot 1: The learned function
plt.subplot(1, 3, 1)
plt.title("Learned function vs Ground truth vs Noisy data")
plt.scatter(x_data, y_data, alpha=0.2, label="Noisy Data")
plt.plot(x_test.cpu().numpy(), np.sin(x_test.cpu().numpy()), 'g-', lw=3, label="True sin(x)")
plt.plot(x_test.cpu().numpy(), y_pred, 'r--', lw=3, label="Actor's Prediction")
plt.xlabel("x (State)")
plt.ylabel("y (Action/Prediction)")
plt.legend(loc="upper left")
plt.grid(True)

# Plot 2: Training Losses
plt.subplot(1, 3, 2)
plt.title("Training Losses")
plt.plot(actor_losses, label="Actor Loss")
plt.plot(critic_losses, label="Critic Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.grid(True)

# Plot 3: Final Error Distribution
plt.subplot(1, 3, 3)
plt.title("Test prediction error vs state (x)")
plt.scatter(x_test, final_errors, alpha=0.5, c=final_errors, cmap='viridis')
plt.xlabel("x (State)"), plt.ylabel("Absolute Error |y - y_hat|")
plt.colorbar(label="Error Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()

"""# Stage 2: explore more in low reward regions."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 1. Setup, Seeds, and Hyperparameters ---
# Set a seed for reproducibility
SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For full reproducibility on CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
EPOCHS = 500
BATCH_SIZE = 64
N_SAMPLES = 1000
SIGMA = 0.2  # Sigma for the Gaussian reward kernel
NOISE_STD = 0.1 # Exploration noise
# How many training steps to perform per epoch
STEPS_PER_EPOCH = int(N_SAMPLES / BATCH_SIZE)

# --- 2. Create the Synthetic Dataset ---
# The agent needs to learn the function y = sin(x)
x_data = np.random.uniform(-5*np.pi, 5*np.pi, N_SAMPLES)
y_data = np.sin(x_data) + np.random.normal(0, 0.1, N_SAMPLES) # Add some noise

# Convert to PyTorch tensors
X = torch.from_numpy(x_data).float().unsqueeze(1).to(device)
Y = torch.from_numpy(y_data).float().unsqueeze(1).to(device)

# --- 3. Prioritized Replay Buffer ---
# This buffer will store experiences and prioritize those with high error.
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha # Controls the level of prioritization
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def add(self, state, true_y):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, true_y))
        else:
            self.buffer[self.pos] = (state, true_y)
        # Set max priority for new experiences to ensure they get sampled
        self.priorities[self.pos] = self.priorities.max() if len(self.buffer) > 1 else 1.0
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None, None

        # Calculate sampling probabilities based on priorities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices based on the calculated probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Unpack the sampled experiences
        states = torch.cat([self.buffer[i][0] for i in indices])
        true_ys = torch.cat([self.buffer[i][1] for i in indices])

        return states, true_ys, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            # Add a small constant to ensure no priority is zero
            self.priorities[idx] = np.abs(error) + 1e-5

    def __len__(self):
        return len(self.buffer)

# --- 4. Define Actor and Critic Networks ---
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh()
        )
    def forward(self, state): return self.net(state)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

# --- 5. Training with Prioritized Replay ---
actor = Actor().to(device)
critic = Critic().to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)
critic_criterion = nn.MSELoss()

# Initialize and populate the replay buffer
replay_buffer = PrioritizedReplayBuffer(N_SAMPLES)
for i in range(N_SAMPLES):
    replay_buffer.add(X[i].unsqueeze(0), Y[i].unsqueeze(0))

actor_losses = []
critic_losses = []

print("Starting training with prioritized exploration...")
for epoch in range(EPOCHS):
    for step in range(STEPS_PER_EPOCH):
        # Sample a batch from the buffer based on priority
        states, true_ys, indices = replay_buffer.sample(BATCH_SIZE)

        # --- Update Critic ---
        actions = actor(states)
        noise = (torch.randn_like(actions) * NOISE_STD).to(device)
        noisy_actions = torch.clamp(actions + noise, -1, 1)
        rewards = torch.exp(-(true_ys - noisy_actions).pow(2) / (2 * SIGMA**2)).detach()
        q_values = critic(states, noisy_actions)
        critic_loss = critic_criterion(q_values, rewards)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # --- Update Actor ---
        actor_loss = -critic(states, actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # --- Update Priorities in the Buffer ---
        # Calculate the new errors for the sampled batch to update their priority
        with torch.no_grad():
            new_actions = actor(states)
            errors = (true_ys - new_actions).squeeze().cpu().numpy()
            replay_buffer.update_priorities(indices, errors)

    actor_losses.append(actor_loss.item())
    critic_losses.append(critic_loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

print("Training finished.")

# --- 6. Visualization ---
x_test = torch.linspace(-10*np.pi, 10*np.pi, 300).unsqueeze(1).to(device)
with torch.no_grad():
    y_pred = actor(x_test).cpu().numpy()

# Create a plot showing final errors across the state space
final_errors = []
with torch.no_grad():
    all_preds = actor(x_test)
    final_errors = (np.sin(x_test) - all_preds).abs().cpu().numpy().flatten()

plt.figure(figsize=(18, 5))

# Plot 1: The learned function
plt.subplot(1, 3, 1)
plt.title("Learned function vs Ground truth vs Noisy data")
plt.scatter(x_data, y_data, alpha=0.2, label="Noisy Data")
plt.plot(x_test.cpu().numpy(), np.sin(x_test.cpu().numpy()), 'g-', lw=3, label="True sin(x)")
plt.plot(x_test.cpu().numpy(), y_pred, 'r--', lw=3, label="Actor's Prediction")
plt.xlabel("x (State)"), plt.ylabel("y (Action/Prediction)"), plt.legend(loc='lower left'), plt.grid(True)

# Plot 2: Training Losses
plt.subplot(1, 3, 2)
plt.title("Training Losses")
plt.plot(actor_losses, label="Actor Loss"), plt.plot(critic_losses, label="Critic Loss")
plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.legend(), plt.grid(True)

# Plot 3: Final Error Distribution
plt.subplot(1, 3, 3)
plt.title("Test prediction error vs state (x)")
plt.scatter(x_test, final_errors, alpha=0.5, c=final_errors, cmap='viridis')
plt.xlabel("x (State)"), plt.ylabel("Absolute Error |y - y_hat|")
plt.colorbar(label="Error Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()

"""# Stage 3: make the Actor and Critic networks deeper and wider."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 1. Setup, Seeds, and Hyperparameters ---
# Set a seed for reproducibility
SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For full reproducibility on CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
EPOCHS = 500
BATCH_SIZE = 64
N_SAMPLES = 1000
SIGMA = 0.2  # Sigma for the Gaussian reward kernel
NOISE_STD = 0.1 # Exploration noise
# How many training steps to perform per epoch
STEPS_PER_EPOCH = int(N_SAMPLES / BATCH_SIZE)

# --- 2. Create the Synthetic Dataset ---
# The agent needs to learn the function y = sin(x) over a wider range
x_data = np.random.uniform(-5*np.pi, 5*np.pi, N_SAMPLES)
y_data = np.sin(x_data) + np.random.normal(0, 0.1, N_SAMPLES) # Add some noise

# Convert to PyTorch tensors
X = torch.from_numpy(x_data).float().unsqueeze(1).to(device)
Y = torch.from_numpy(y_data).float().unsqueeze(1).to(device)

# --- 3. Prioritized Replay Buffer ---
# This buffer will store experiences and prioritize those with high error.
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha # Controls the level of prioritization
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def add(self, state, true_y):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, true_y))
        else:
            self.buffer[self.pos] = (state, true_y)
        self.priorities[self.pos] = self.priorities.max() if len(self.buffer) > 1 else 1.0
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0: return None, None, None
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        states = torch.cat([self.buffer[i][0] for i in indices])
        true_ys = torch.cat([self.buffer[i][1] for i in indices])
        return states, true_ys, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = np.abs(error) + 1e-5

    def __len__(self):
        return len(self.buffer)

# --- 4. Define Actor and Critic Networks (IMPROVED) ---
# The networks are now deeper and wider to handle the more complex function.
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), # Extra hidden layer
            nn.Linear(64, 1), nn.Tanh()
        )
    def forward(self, state): return self.net(state)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), # Extra hidden layer
            nn.Linear(64, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

# --- 5. Training with Prioritized Replay ---
actor = Actor().to(device)
critic = Critic().to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)
critic_criterion = nn.MSELoss()

replay_buffer = PrioritizedReplayBuffer(N_SAMPLES)
for i in range(N_SAMPLES):
    replay_buffer.add(X[i].unsqueeze(0), Y[i].unsqueeze(0))

actor_losses = []
critic_losses = []

print("Starting training with IMPROVED networks...")
for epoch in range(EPOCHS):
    for step in range(STEPS_PER_EPOCH):
        states, true_ys, indices = replay_buffer.sample(BATCH_SIZE)

        # --- Update Critic ---
        actions = actor(states)
        noise = (torch.randn_like(actions) * NOISE_STD).to(device)
        noisy_actions = torch.clamp(actions + noise, -1, 1)
        rewards = torch.exp(-(true_ys - noisy_actions).pow(2) / (2 * SIGMA**2)).detach()
        q_values = critic(states, noisy_actions)
        critic_loss = critic_criterion(q_values, rewards)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # --- Update Actor ---
        actor_loss = -critic(states, actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # --- Update Priorities in the Buffer ---
        with torch.no_grad():
            new_actions = actor(states)
            errors = (true_ys - new_actions).squeeze().cpu().numpy()
            replay_buffer.update_priorities(indices, errors)

    actor_losses.append(actor_loss.item())
    critic_losses.append(critic_loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

print("Training finished.")

# --- 6. Visualization ---
x_test = torch.linspace(-10*np.pi, 10*np.pi, 300).unsqueeze(1).to(device)
with torch.no_grad():
    y_pred = actor(x_test).cpu().numpy()

# Create a plot showing final errors across the state space
final_errors = []
with torch.no_grad():
    all_preds = actor(x_test)
    final_errors = (np.sin(x_test) - all_preds).abs().cpu().numpy().flatten()

plt.figure(figsize=(18, 5))

# Plot 1: The learned function
plt.subplot(1, 3, 1)
plt.title("Learned function vs Ground truth vs Noisy data")
plt.scatter(x_data, y_data, alpha=0.2, label="Noisy Data")
plt.plot(x_test.cpu().numpy(), np.sin(x_test.cpu().numpy()), 'g-', lw=3, label="True sin(x)")
plt.plot(x_test.cpu().numpy(), y_pred, 'r--', lw=3, label="Actor's Prediction")
plt.xlabel("x (State)"), plt.ylabel("y (Action/Prediction)"), plt.legend(loc='lower left'), plt.grid(True)

# Plot 2: Training Losses
plt.subplot(1, 3, 2)
plt.title("Training Losses")
plt.plot(actor_losses, label="Actor Loss"), plt.plot(critic_losses, label="Critic Loss")
plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.legend(), plt.grid(True)

# Plot 3: Final Error Distribution
plt.subplot(1, 3, 3)
plt.title("Test prediction error vs state (x)")
plt.scatter(x_test, final_errors, alpha=0.5, c=final_errors, cmap='viridis')
plt.xlabel("x (State)"), plt.ylabel("Absolute Error |y - y_hat|")
plt.colorbar(label="Error Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()

"""# Stage 4: Add Positional Encoding"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 1. Setup, Seeds, and Hyperparameters ---
# Set a seed for reproducibility
SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For full reproducibility on CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
EPOCHS = 500
BATCH_SIZE = 64
N_SAMPLES = 1000
SIGMA = 0.2  # Sigma for the Gaussian reward kernel
NOISE_STD = 0.1 # Exploration noise
STEPS_PER_EPOCH = int(N_SAMPLES / BATCH_SIZE)
ENCODING_DIMS = 16 # Number of dimensions for positional encoding

# --- 2. Positional Encoding ---
# This function transforms the input 'x' into a richer feature vector
def positional_encoding(x, n_dims):
    if x.dim() == 1:
        x = x.unsqueeze(1)

    device = x.device
    # Create frequencies from 0 to n_dims/2
    freqs = 2.0**torch.arange(n_dims / 2, device=device)
    # Create the arguments for sin and cos
    args = x * freqs
    # Concatenate sin and cos encodings
    encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return encoding

# The total dimension of the state after encoding
state_dim = ENCODING_DIMS

# --- 3. Create the Synthetic Dataset ---
x_data_raw = np.random.uniform(-5*np.pi, 5*np.pi, N_SAMPLES)
y_data = np.sin(x_data_raw) + np.random.normal(0, 0.1, N_SAMPLES)

# Convert to PyTorch tensors
X_raw = torch.from_numpy(x_data_raw).float().unsqueeze(1).to(device)
Y = torch.from_numpy(y_data).float().unsqueeze(1).to(device)

# Apply positional encoding to the state features
X_encoded = positional_encoding(X_raw, ENCODING_DIMS)

# --- 4. Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def add(self, state, true_y):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, true_y))
        else:
            self.buffer[self.pos] = (state, true_y)
        self.priorities[self.pos] = self.priorities.max() if len(self.buffer) > 1 else 1.0
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0: return None, None, None
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        states = torch.cat([self.buffer[i][0] for i in indices])
        true_ys = torch.cat([self.buffer[i][1] for i in indices])
        return states, true_ys, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = np.abs(error) + 1e-5

    def __len__(self):
        return len(self.buffer)

# --- 5. Define Actor and Critic Networks (Updated for Encoded State) ---
class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), # Input layer accepts encoded state
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh()
        )
    def forward(self, state): return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        # The input dimension is state_dim + 1 (for the action)
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

# --- 6. Training with Positional Encoding ---
actor = Actor(state_dim).to(device)
critic = Critic(state_dim).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)
critic_criterion = nn.MSELoss()

replay_buffer = PrioritizedReplayBuffer(N_SAMPLES)
for i in range(N_SAMPLES):
    replay_buffer.add(X_encoded[i].unsqueeze(0), Y[i].unsqueeze(0))

actor_losses, critic_losses = [], []

print("Starting training with Positional Encoding...")
for epoch in range(EPOCHS):
    for step in range(STEPS_PER_EPOCH):
        states, true_ys, indices = replay_buffer.sample(BATCH_SIZE)

        actions = actor(states)
        noise = (torch.randn_like(actions) * NOISE_STD).to(device)
        noisy_actions = torch.clamp(actions + noise, -1, 1)
        rewards = torch.exp(-(true_ys - noisy_actions).pow(2) / (2 * SIGMA**2)).detach()
        q_values = critic(states, noisy_actions)
        critic_loss = critic_criterion(q_values, rewards)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actor_loss = -critic(states, actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        with torch.no_grad():
            errors = (true_ys - actor(states)).squeeze().cpu().numpy()
            replay_buffer.update_priorities(indices, errors)

    actor_losses.append(actor_loss.item())
    critic_losses.append(critic_loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

print("Training finished.")

# --- 7. Visualization ---
# Generate a clean set of x values for plotting the learned function
x_test_raw = torch.linspace(-10*np.pi, 10*np.pi, 400).unsqueeze(1).to(device)
x_test_encoded = positional_encoding(x_test_raw, ENCODING_DIMS)
with torch.no_grad():
    y_pred = actor(x_test_encoded).cpu().numpy()

# Create a plot showing final errors across the training data space
final_errors = []
with torch.no_grad():
    # Get predictions for the original training data (using the encoded version)
    all_preds = actor(X_encoded)
    # Calculate the error against the true training labels Y
    final_errors = (Y - all_preds).abs().cpu().numpy().flatten()

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.title("Learned function vs Ground truth vs Noisy data")
plt.scatter(x_data_raw, y_data, alpha=0.2, label="Noisy Data")
plt.plot(x_test_raw.cpu().numpy(), np.sin(x_test_raw.cpu().numpy()), 'g-', lw=3, label="True sin(x)")
plt.plot(x_test_raw.cpu().numpy(), y_pred, 'r--', lw=2, label="Actor's Prediction")
plt.xlabel("x (State)"), plt.ylabel("y (Action/Prediction)"), plt.legend(), plt.grid(True)
plt.ylim(-1.5, 1.5)

plt.subplot(1, 3, 2)
plt.title("Training Losses")
plt.plot(actor_losses, label="Actor Loss"), plt.plot(critic_losses, label="Critic Loss")
plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.legend(), plt.grid(True)

# Plot 3: Final Error Distribution
plt.subplot(1, 3, 3)
plt.title("Final Prediction Error vs. State (x)")
# Use the raw x-values from the training data for the scatter plot
plt.scatter(x_data_raw, final_errors, alpha=0.5, c=final_errors, cmap='viridis')
plt.xlabel("x (State)"), plt.ylabel("Absolute Error |y - y_hat|")
plt.colorbar(label="Error Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()
