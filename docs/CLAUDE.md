# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of the paper "Solving 3D packing problem using Transformer network and reinforcement learning" (https://www.sciencedirect.com/science/article/abs/pii/S0957417422021716). The system uses PPO (Proximal Policy Optimization) to train a Transformer-based policy network for solving the 3D bin packing problem.

## Common Commands

### Conda 
```bash
conda activate torch310
```

### Training
```bash
# Train the model with default parameters
python train.py

# Train with custom configuration
python train.py --config config.yaml

# View all training options
python train.py --help

# Key training parameters:
# --length: Container length (default: 100)
# --width: Container width (default: 100)
# --num_boxes: Number of boxes to sample (default: 50)
# --epochs: Number of training epochs (default: 1,000,000)
# --ppo_epochs: Number of PPO sub-epochs per episode (default: 17)
# --beta: Entropy regularization coefficient (default: 0.01)
# --save_path: Directory to save trained models (default: saved_models)
```

### Testing
```bash
# Test a trained model (requires policy model path)
python test.py --policy_model_path saved_models/ppo_policy.pth

# Test with custom configuration
python test.py --config config.yaml --policy_model_path saved_models/ppo_policy.pth

# View all testing options
python test.py --help
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Core Components

**Environment (`envs/packingEnv.py`)**
- `PackingEnv`: Gymnasium-based environment for 3D bin packing
- State representation:
  - Container state: `(L, W, 7)` grid with height map + directional distance features
  - Unpacked boxes state: `(N, 3)` array of box dimensions (length, width, height)
- Action space: Dict with position `(x, y)`, box selection index, and orientation (6 possibilities)
- Reward: Reduction in unused volume (g-value) minus step penalty; large penalty for invalid placements
- Rendering: Supports 'human', 'rgb_array', and 'ansi' modes; can save packing process as GIF

**Policy Network (`models/policy.py`)**
- `PolicyNetwork`: Transformer-based actor network that outputs actions in 3 stages:
  1. Position selection (where to place)
  2. Box selection (which box to place)
  3. Orientation selection (how to orient the box)
- Uses separate encoder-decoder architecture for each action component
- Returns actions, log probabilities, and probability distributions (needed for PPO)

**Value Network (`models/value.py`)**
- `ValueNetwork`: Transformer-based critic network for state value estimation
- Uses learned query token that attends over concatenated box and container encodings
- Outputs scalar state value V(s_t) for PPO advantage calculation
- Does NOT share parameters with policy network to avoid update interference

**Encoders (`models/encoders.py`)**
- `BoxEncoder`: Encodes unpacked boxes (N boxes × 3 dimensions) into sequence of embeddings
- `ContainerEncoder`: Encodes container state using patch-based approach:
  - Downsamples container grid into patches (default: 10×10 patches for 100×100 container)
  - For each patch, selects representative cell with max(right_distance × down_distance)
  - Applies transformer encoding with positional embeddings
  - Returns encoded sequence and global (i,j) indices for selected cells

**Decoders (`models/decoders.py`)**
- `PositionDecoder`: Attends over box encoding (memory) with container encoding (query) to select placement position
- `SelectionDecoder`: Attends over position embedding to select which box to place; generates 6 orientation embeddings for chosen box
- `OrientationDecoder`: Attends over position embedding to select box orientation from 6 possibilities

### Training Flow (`train.py`)

1. **Trajectory Collection** (`collect_trajectory`):
   - Run episode with current policy up to max_steps (default: 250)
   - For each step: get action from policy, execute in environment, store (state, action, reward, value, next_value, done, probs)
   - Episode terminates when all boxes placed or max steps reached

2. **Advantage Computation** (`compute_returns_and_advantages`):
   - Uses Generalized Advantage Estimation (GAE) with γ=0.99, λ=0.96
   - Computes TD residuals: δ = r + γV(s') - V(s)
   - Accumulates advantages via GAE formula
   - Normalizes returns and advantages for stability

3. **PPO Loss** (`compute_ppo_loss`):
   - Actor loss: Clipped surrogate objective with ε=0.12
   - Critic loss: MSE between predicted values and computed returns
   - Entropy loss: Weighted by β (default: 0.01) for exploration
   - Total loss = actor_loss + critic_loss + β × entropy_loss

4. **Training Loop** (`train_ppo`):
   - For each epoch:
     - Generate random box dimensions (l, w, h) based on container size
     - Collect trajectory with current policy
     - Run PPO updates for ppo_epochs (default: 27) sub-iterations
     - Update both policy and value networks
   - Saves trained models to `saved_models/` directory

### Key Implementation Details

- **Device Selection**: Automatically uses CUDA if available, else MPS (Apple Silicon), else CPU
- **Box Orientations**: 6 possible orientations generated by permuting (l, w, h) dimensions
- **Container Features**: Each cell has 7 features:
  1. Current height
  2-5. Directional distances (right, down, left, up) to different height
  6-7. Next-level distances (right_next, down_next) to higher cells
- **Patch Downsampling**: Reduces 100×100 grid to 10×10 patches (100 tokens) for computational efficiency
- **Training Duration**: README notes training takes 4 days to a week for passable results
- **Invalid Placements**: Receive large penalty (-10^7) and don't update container state

### Architecture Diagram Flow

The policy network follows this sequence:
1. BoxEncoder + ContainerEncoder process raw states → encoded sequences
2. PositionDecoder: container_encoding (query) attends to box_encoding (memory) → position action + position_embedding
3. SelectionDecoder: box_encoding (query) attends to position_embedding (memory) → box selection + box_orientation_embedding (6 orientations)
4. OrientationDecoder: box_orientation_embedding (query) attends to position_embedding (memory) → orientation action

The value network:
1. BoxEncoder + ContainerEncoder process raw states
2. Learned query token attends to concatenated [box_tokens, container_tokens]
3. MLP head outputs scalar value

### Model Persistence

- Policy network saved as: `saved_models/ppo_policy.pth`
- Value network saved as: `saved_models/ppo_value.pth`
- Test script loads policy network only (value network not needed for inference)