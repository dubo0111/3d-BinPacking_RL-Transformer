import os
import random
from torch import optim
import torch.nn.functional as F
from envs.packingEnv import *
from models.policy import *
from models.value import *
import argparse
import yaml

DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

def collect_trajectory(policy_net, value_net, env, max_steps=250, device=DEVICE_STR):
    obs, info = env.reset()  # Reset environment
    container_state = obs['container_state']
    unpacked_boxes_state = env.unpacked_boxes
    trajectory = []

    for _ in range(max_steps):
        container_state_T = torch.tensor(container_state, dtype=torch.float32).unsqueeze(0).to(device)
        unpacked_boxes_state_T = torch.tensor(unpacked_boxes_state, dtype=torch.float32).unsqueeze(0).to(device)
        action, log_prob_action, prob_all = policy_net(container_state_T, unpacked_boxes_state_T, deterministic_selection=False)
        value = value_net(container_state_T, unpacked_boxes_state_T)

        observation, reward, terminated, truncated, info = env.step({
            "position": action[0],
            "box_select": action[1].item(),
            "orientation": action[2].item()
        })

        
        # Get value of the next state (0 if done)
        next_container_state = observation['container_state']
        next_unpacked_boxes_state = env.unpacked_boxes

        next_container_state_T = torch.tensor(next_container_state, dtype=torch.float32).unsqueeze(0).to(device)
        next_unpacked_boxes_state_T = torch.tensor(next_unpacked_boxes_state, dtype=torch.float32).unsqueeze(0).to(device)

        next_value = value_net(next_container_state_T, next_unpacked_boxes_state_T) if not terminated else torch.tensor(0.0)
        if len(unpacked_boxes_state) == 0:
            unpacked_boxes_state = [(0, 0, 0)]
        # Store trajectory data
        trajectory.append((container_state, 
                           unpacked_boxes_state.copy(), 
                           (action[0], action[1].item(), action[2].item()), 
                           reward, 
                           (log_prob_action[0], log_prob_action[1], log_prob_action[2]), 
                           value.item(), 
                           next_value.item(), 
                           terminated, 
                           prob_all))

        if terminated:
            break
        container_state = next_container_state
        unpacked_boxes_state = next_unpacked_boxes_state

    return trajectory

def compute_returns_and_advantages(trajectory, gamma=0.99, lamb=0.96, device=DEVICE_STR):
    gae = 0
    advantages = []
    returns = []
    
    for step in reversed(trajectory):
        _, _, _, reward, _, value, next_value, done, _ = step
        
        delta = reward + gamma * (next_value if not done else 0) - value
        
        gae = delta + (gamma*lamb) * gae * (1 - int(done)) # refer above
        
        return_t = gae + value
        returns.insert(0, return_t)
        advantages.insert(0, gae)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def compute_ppo_loss(policy_net, value_net, container_states, unpacked_boxes_states, old_probs, advantages, returns, epsilon=0.12, beta=0.01, device=DEVICE_STR):
    policy_output = [policy_net(container_states, unpacked_boxes_states, deterministic_selection=False) for 
                     container_states, unpacked_boxes_states in zip(container_states, unpacked_boxes_states)]
    new_log_probs = [output[1] for output in policy_output]
    new_probs = [output[2] for output in policy_output]




    new_probs_tensor = torch.stack([torch.stack([lp for lp in lps]) if isinstance(lps, (list, tuple)) else lps for lps in new_probs])
    ratio = new_probs_tensor.prod(dim=1) / old_probs

    values = torch.stack([value_net(container_states, unpacked_boxes_states) for 
              container_states, unpacked_boxes_states in zip(container_states, unpacked_boxes_states)]).squeeze().to(device)

    surr1 = ratio*advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    

    critic_loss = F.mse_loss(values, returns)  # Mean Squared Error loss for critic
    

    probs_tensor = torch.stack([torch.tensor(t, dtype=torch.float32) for t in new_probs]).to(device)  # shape (n, 3)
    log_probs_tensor = torch.stack([torch.tensor(t, dtype=torch.float32) for t in new_log_probs]).to(device)  # shape (n, 3)
    # Elementwise multiply and sum over last dimension, then sum over all samples
    entropy_loss = -torch.sum(probs_tensor * log_probs_tensor)

    total_loss = actor_loss + critic_loss + beta * entropy_loss
    return actor_loss, critic_loss, beta*entropy_loss, total_loss


def train_ppo(env_params, policy_net, value_net, num_epochs=1, max_steps=250, ppo_epochs=27, device=DEVICE_STR, beta=0.01):
    # Declaring optimizers for both nets
    L, W, n = env_params
    container_dims = (L, W)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-5)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        
        l_samples = [random.randint(L // 10, L // 2) for _ in range(n)]
        w_samples = [random.randint(W // 10, W // 2) for _ in range(n)]
        h_samples = [random.randint(min(L, W) // 10, max(L, W) // 2) for _ in range(n)]

        boxes = list(zip(l_samples, w_samples, h_samples))

        env = PackingEnv(container_dims=container_dims, initial_boxes=boxes, render_mode=None)
        obs, info = env.reset()
        trajectory = collect_trajectory(policy_net, value_net, env, max_steps) 
        states=[(state[0], state[1]) for state in trajectory]

        returns, advantages = compute_returns_and_advantages(trajectory)

        container_states = [torch.tensor(state[0]).unsqueeze(0).to(device) for state in states]
        unpacked_boxes_states = [torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(device) for state in states]


        old_log_probs = torch.tensor([s[4] for s in trajectory], dtype=torch.float32).to(device)
        old_probs = torch.tensor([s[-1] for s in trajectory], dtype=torch.float32).to(device).prod(dim=1)
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_loss = 0
        for _ in range(ppo_epochs):
            actor_loss, critic_los, beta_entropy, loss = compute_ppo_loss(policy_net, value_net, container_states, unpacked_boxes_states, old_probs, advantages, returns, beta=beta, device=device)
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_los.item()
            total_entropy += beta_entropy.item()
            total_loss += loss.item()
            #print(f"Actor Loss: {actor_loss.item()}, Critic Loss: {critic_los.item()}, Entropy Loss: {beta_entropy.item()}, total Loss: {loss.item()}")
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()
            value_optimizer.step()
        total_reward = sum([t[3] for t in trajectory])
        print(f"Epoch {epoch + 1}/{num_epochs}, Total Reward: {total_reward}, Total Actor Loss: {total_actor_loss}, Total Critic Loss: {total_critic_loss}, Total Entropy Loss: {total_entropy}, Total Loss: {total_loss}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO training on packing environment.")
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--length', type=int, default=100, help='Length of the container')
    parser.add_argument('--width', type=int, default=100, help='Width of the container')
    parser.add_argument('--num_boxes', type=int, default=50, help='Number of boxes to sample')
    parser.add_argument('--epochs', type=int, default=1_000_000, help='Number of PPO training epochs')
    parser.add_argument('--ppo_epochs', type=int, default=17, help='Number of PPO sub-epochs')
    parser.add_argument('--beta', type=float, default=0.01, help='Entropy regularization term')
    parser.add_argument('--save_path', type=str, default='saved_models', help='Path to save the trained model')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    L = args.length
    W = args.width
    num_boxes = args.num_boxes
    env_params = (L, W, num_boxes)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    policy_net = PolicyNetwork().to(device)
    value_net = ValueNetwork().to(device)

    train_ppo(env_params, policy_net, value_net,
              num_epochs=args.epochs,
              ppo_epochs=args.ppo_epochs,
              beta=args.beta,
              device=device)
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    torch.save(policy_net.state_dict(), os.path.join(args.save_path, 'ppo_policy.pth'))
    torch.save(value_net.state_dict(), os.path.join(args.save_path, 'ppo_value.pth'))
