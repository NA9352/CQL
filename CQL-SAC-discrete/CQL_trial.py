"""
Behavioral cloning with PyTorch
=========================================
"""
# %%%
# We present here how to perform behavioral cloning on a Minari dataset using `PyTorch <https://pytorch.org/>`_.
# We will start generating the dataset of the expert policy for the `CartPole-v1 <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ environment, which is a classic control problem.
# The objective is to balance the pole on the cart, and we receive a reward of +1 for each successful timestep.
import torch.nn as nn
import torch.nn.functional as F
import minari
from torch.utils.data import DataLoader
import gymnasium as gym
from gymnasium import spaces
from agent import CQLSAC
import numpy as np
import torch

# %%
# In this scenario, the output dimension will be two, as previously mentioned. As for the input dimension, it will be four, corresponding to the observation space of ``CartPole-v1``.
# Our next step is to load the dataset and set up the training loop. The ``MinariDataset`` is compatible with the PyTorch Dataset API, allowing us to load it directly using `PyTorch DataLoader <https://pytorch.org/docs/stable/data.html>`_.
# However, since each episode can have a varying length, we need to pad them.
# To achieve this, we can utilize the `collate_fn <https://pytorch.org/docs/stable/data.html#working-with-collate-fn>`_ feature of PyTorch DataLoader. Let's create the ``collate_fn`` function:
def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "seed": torch.Tensor([x.seed for x in batch]),
        "total_timesteps": torch.Tensor([x.total_timesteps for x in batch]),
        "observations": {
            'image': torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.observations['image']) for x in batch],
                batch_first=True
            ),
            'direction': torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.observations['direction']) for x in batch],
                batch_first=True
            ),
        },
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch],
            batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch],
            batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch],
            batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch],
            batch_first=True
        )
    }
# %%
# We can now proceed to load the data and create the training loop.
# To begin, let's initialize the DataLoader, neural network, optimizer, and loss.
minari_dataset = minari.load_dataset("minigrid-fourrooms-v0")
dataloader = DataLoader(minari_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
env = minari_dataset.recover_environment()

env.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8),
            'direction': spaces.Discrete(4), })
observation_space = env.observation_space
action_space = env.action_space

for key, space in observation_space.items():
    assert isinstance(space, (spaces.Box, spaces.Discrete)), f"{key} space is not a Box or Discrete space"
assert isinstance(action_space, spaces.Discrete)

# Calculate the total size of the observation space
obs_space_size = np.prod(observation_space['image'].shape) + 1 #observation_space['direction'].n
agent = CQLSAC(obs_space_size, action_space.n,"cpu")

# %%
num_epochs = 20

# In your training loop
for epoch in range(num_epochs):
    # Learn from a batch of experiences
    for batch in dataloader:

        # Extract the 'image' and 'direction' data from the observations
        image = batch['observations']['image']
        direction = batch['observations']['direction']

        # Flatten the color channel dimension of the image tensor
        image_flattened = image.reshape(image.shape[0], image.shape[1], -1).float()

        # Add an extra dimension to the direction tensor to match the image tensor
        direction_expanded = direction.unsqueeze(-1).float()

        # Concatenate the image and direction tensors along the last dimension
        observations = torch.cat((image_flattened, direction_expanded), dim=-1).float()

        experiences = (observations[:,:-1], batch['actions'], batch['rewards'].float(), observations[:,1:] ,batch['terminations'].float())

        policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn(epoch, experiences, gamma=0.99)
    print(f"Epoch {epoch} - Policy Loss: {policy_loss}, Alpha Loss: {alpha_loss}, Bellmann Error 1: {bellmann_error1}, Bellmann Error 2: {bellmann_error2}, CQL1 Loss: {cql1_loss}, CQL2 Loss: {cql2_loss}, Current Alpha: {current_alpha}, Lagrange Alpha Loss: {lagrange_alpha_loss}, Lagrange Alpha: {lagrange_alpha}")

# %%
render = True
env = gym.make("MiniGrid-FourRooms-v0")
accumulated_rew = 0
n_runs = 1
softmax = nn.Softmax(dim=0)
from random import seed

for i in range(n_runs):
    images = []
    n_steps = 0
    obs, _ = env.reset(seed=seed())
    done = False
    while not done:
        # Process the observations
        obs_input = np.concatenate((obs['image'].flatten(), np.array([obs['direction']])))

        # action = policy_net(torch.Tensor(obs_input)).argmax().item()
        action = torch.as_tensor(agent.get_action(obs_input))
        obs, rew, ter, tru, _ = env.step(action)
        done = ter or tru
        accumulated_rew += rew
        n_steps += 1
        if render:
            images.append(env.get_full_render(True, 16))

    # print(f"{'Terminated' if ter else 'Truncated'} in {n_steps} steps.")

env.close()
print("Average reward: ", accumulated_rew / n_runs)
# %%
if render:
    from array2gif import write_gif
    write_gif(images, 'last_episode.gif', fps=5)

# %% Code trying to replicate the CQLSAC class
# Critic Network (w/ Target Network)

from networks import Critic, Actor
import torch
import torch.optim as optim

state_size = obs_space_size
action_size = action_space.n
hidden_size = 256
device = "cpu"
learning_rate = 5e-4
next_states = observations[:,1:]
with_lagrange = False

critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)

assert critic1.parameters() != critic2.parameters()

critic1_target = Critic(state_size, action_size, hidden_size).to(device)
critic1_target.load_state_dict(critic1.state_dict())

critic2_target = Critic(state_size, action_size, hidden_size).to(device)
critic2_target.load_state_dict(critic2.state_dict())

critic1_optimizer = optim.Adam(critic1.parameters(), lr=learning_rate)
critic2_optimizer = optim.Adam(critic2.parameters(), lr=learning_rate) 
softmax = nn.Softmax(dim=-1)

actor_local = Actor(state_size, action_size, hidden_size).to(device)

log_alpha = torch.tensor([0.0], requires_grad=True)
alpha = log_alpha.exp().detach()

with torch.no_grad():
    _, action_probs, log_pis = actor_local.evaluate(next_states)
    Q_target1_next = critic1_target(next_states)
    Q_target2_next = critic2_target(next_states)
    Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - alpha.to(device) * log_pis)

    # Compute Q targets for current states (y_i)
    Q_targets = batch['rewards'].float() + (0.99 * (1 - batch['terminations'].float()) * Q_target_next.sum(dim=2))

# Compute critic loss
q1 = critic1(observations[:,:-1])
q2 = critic2(observations[:,:-1])

q1_ = q1.gather(1, batch['actions'].unsqueeze(-1).long())
q2_ = q2.gather(1, batch['actions'].unsqueeze(-1).long())

critic1_loss = 0.5 * F.mse_loss(q1_, Q_targets.unsqueeze(-1))
critic2_loss = 0.5 * F.mse_loss(q2_, Q_targets.unsqueeze(-1))

cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()

cql_alpha_loss = torch.FloatTensor([0.0])
cql_alpha = torch.FloatTensor([0.0])

total_c1_loss = critic1_loss + cql1_scaled_loss
total_c2_loss = critic2_loss + cql2_scaled_loss

critic1_optimizer.zero_grad()
total_c1_loss = total_c1_loss
total_c1_loss.backward(retain_graph=True)


