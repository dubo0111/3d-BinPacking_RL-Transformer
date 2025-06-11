import os
import random
import argparse
import yaml
from torch import optim
import torch.nn.functional as F
from envs.packingEnv import *
from models.policy import *
from models.value import *

DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

def test_agent(env, policy_net, max_steps=500, device=DEVICE_STR):
    obs, info = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        container_state_T = torch.tensor(obs['container_state'], dtype=torch.float32).unsqueeze(0).to(device)
        unpacked_boxes_state_T = torch.tensor(env.unpacked_boxes, dtype=torch.float32).unsqueeze(0).to(device)
        action, _ , __annotations__ = policy_net(container_state_T, unpacked_boxes_state_T, deterministic_selection=False)

        obs, reward, terminated, truncated, info = env.step({
            "position": action[0],
            "box_select": action[1].item(),
            "orientation": action[2].item()
        })

        total_reward += reward
        if terminated:
            break
        print(f"Test Total Reward: {total_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained agent in the packing environment.")
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--length', type=int, default=100, help='Length of the container')
    parser.add_argument('--width', type=int, default=100, help='Width of the container')
    parser.add_argument('--num_boxes', type=int, default=50, help='Number of boxes to sample')
    parser.add_argument('--policy_model_path', type=str, required=True, help='Path to the trained policy model.')
    args = parser.parse_args()

    # If a YAML config is provided, load and update arguments
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    container_dims = (args.length, args.width)
    L, W = container_dims
    n = args.num_boxes

    l_samples = [random.randint(L // 10, L // 2) for _ in range(n)]
    w_samples = [random.randint(W // 10, W // 2) for _ in range(n)]
    h_samples = [random.randint(min(L, W) // 10, max(L, W) // 2) for _ in range(n)]

    boxes = list(zip(l_samples, w_samples, h_samples))

    env = PackingEnv(container_dims=container_dims, initial_boxes=boxes, render_mode='human')
    obs, info = env.reset()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    policy_net = PolicyNetwork().to(device)
    
    # Load the trained policy model
    policy_net.load_state_dict(torch.load(args.policy_model_path, map_location=DEVICE_STR))
    policy_net.eval()

    test_agent(env, policy_net, device=device)

    env.close()