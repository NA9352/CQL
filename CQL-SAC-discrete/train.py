

import gym
#simport pybullet_envs
import numpy as np
from collections import deque
import torch
#import wandb
import argparse
from buffer import ReplayBuffer
import glob
from utils import save, collect_random
import random
from agent import CQLSAC
import minari
from gymnasium import spaces

def get_config(run_name="MyRun", env="MiniGrid-FourRooms-v0", episodes=500, 
               buffer_size=200000, seed=42, log_video=1, save_every=50,
                 batch_size=512):
    config = argparse.Namespace()
    config.run_name = run_name
    config.env = env
    config.episodes = episodes
    config.buffer_size = buffer_size
    config.seed = seed
    config.log_video = log_video
    config.save_every = save_every
    config.batch_size = batch_size
    return config

#def train(config):
config = get_config()
np.random.seed(config.seed)
random.seed(config.seed)
torch.manual_seed(config.seed)
env = gym.make(config.env)

#env.seed(config.seed)
#env.action_space.seed(config.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

steps = 0
average10 = deque(maxlen=10)
total_steps = 0

observation_space = env.observation_space
action_space = env.action_space
 
for key, space in observation_space.items():
    assert isinstance(space, (spaces.Box, spaces.Discrete)), f"{key} space is not a Box or Discrete space"

assert isinstance(action_space, spaces.Discrete)
env.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8),
            'direction': spaces.Discrete(4), })
observation_space = env.observation_space
action_space = env.action_space

agent = CQLSAC(state_size=np.product(env.observation_space['image'].shape)+1,
                    action_size=env.action_space.n,
                    device=device)

    #wandb.watch(agent, log="gradients", log_freq=10)

buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
    
collect_random(env=env, dataset=buffer, num_samples=1000)

if config.log_video:
    env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

for i in range(1, config.episodes+1):
    state = env.reset()
    episode_steps = 0
    rewards = 0
    while True:
        action = agent.get_action(state)
        steps += 1
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn(steps, buffer.sample(), gamma=0.99)
        state = next_state
        rewards += reward
        episode_steps += 1
        if done:
            break

        

    average10.append(rewards)
    total_steps += episode_steps
    print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))
    
    wandb.log({"Reward": rewards,
                "Average10": np.mean(average10),
                "Steps": total_steps,
                "Policy Loss": policy_loss,
                "Alpha Loss": alpha_loss,
                "Lagrange Alpha Loss": lagrange_alpha_loss,
                "CQL1 Loss": cql1_loss,
                "CQL2 Loss": cql2_loss,
                "Bellmann error 1": bellmann_error1,
                "Bellmann error 2": bellmann_error2,
                "Alpha": current_alpha,
                "Lagrange Alpha": lagrange_alpha,
                "Steps": steps,
                "Episode": i,
                "Buffer size": buffer.__len__()})

    if (i %10 == 0) and config.log_video:
        mp4list = glob.glob('video/*.mp4')
        if len(mp4list) > 1:
            mp4 = mp4list[-2]
            wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

    if i % config.save_every == 0:
        save(config, save_name="CQL-SAC-discrete", model=agent.actor_local, wandb=wandb, ep=0)

if __name__ == "__main__":
    config = get_config()
    train(config)

minari