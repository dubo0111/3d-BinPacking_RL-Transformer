# Paper vs Code Implementation Comparison

This document provides a detailed comparison between the paper "Solving 3D packing problem using Transformer network and reinforcement learning" and its code implementation.

## Executive Summary

The implementation is generally faithful to the paper's methodology with some notable differences in hyperparameters, reward function details, and training procedures. The core architecture (Transformer-based encoder-decoder) and novel contributions (state representation, plane features, action ordering) are correctly implemented.

---

## 1. State Representation

### Paper Description
- **Container State**: `(L×W×7)` matrix where each cell contains:
  - `h_ij`: Current stacked height
  - `e^l, e^w, e^-l, e^-w`: Directional distances to edges with same height
  - `f^l, f^w`: Distances to nearest higher plane
- **Box State**: `{b_1, b_2, ..., b_n}` where `b_i = (l_i, w_i, h_i)` (unpacked boxes only)
- **Key Innovation**: Separating container and box states to avoid multiple states representing same container configuration

### Code Implementation
**Match**: ✅ Correctly implemented

```python
# envs/packingEnv.py:126-139
def _calculate_plane_features(self):
    plane_features_map = np.zeros((self.container_L, self.container_W, 7), dtype=np.float32)
    for r in range(self.container_L):
        for c in range(self.container_W):
            current_h = self.container_height_map[r, c]
            plane_features_map[r, c, 0] = current_h
            features = self.directional_distance(r, c)
            plane_features_map[r, c, 1] = features['right']
            plane_features_map[r, c, 2] = features['down']
            plane_features_map[r, c, 3] = features['left']
            plane_features_map[r, c, 4] = features['up']
            plane_features_map[r, c, 5] = features['right_next']
            plane_features_map[r, c, 6] = features['down_next']
```

**Notes**:
- Directional distance calculation matches paper's Fig. 2
- The implementation uses `right`, `down`, `left`, `up` which map to paper's `e^l, e^w, e^-l, e^-w`
- `right_next` and `down_next` map to paper's `f^l, f^w`

---

## 2. Action Space Decomposition

### Paper Description
- **Action Order**: Position → Selection → Orientation
- **Formula**: `π(a_t|s_t) = π(a_t^p|s_t) × π(a_t^s|a_t^p,s_t) × π(a_t^o|a_t^p,a_t^s,s_t)`
- **Rationale**: More intuitive for large box counts (decide where first, then what/how)

### Code Implementation
**Match**: ✅ Correctly implemented

```python
# models/policy.py:78-99
# Position Decoding
chosen_pos_idx, log_prob_chosen_pos, position_embedding, position_probs_all = self.position_decoder(...)

# Selection Decoding (depends on position_embedding)
chosen_box_idx, log_prob_chosen_box, box_orientation_embedding, box_selection_probs_all = self.selection_decoder(
    box_encoding=box_encoding,
    position_embedding=position_embedding, ...
)

# Orientation Decoding (depends on both position and box selection)
chosen_orient_idx, log_prob_chosen_orient, orientation_probs_all = self.orientation_decoder(
    box_orientation_embedding=box_orientation_embedding,
    position_embedding=position_embedding, ...
)
```

**Notes**: The sequential dependency chain is properly implemented.

---

## 3. Container Downsampling

### Paper Description
- **Method**: Partition `100×100×7` container into `10×10` patches
- **Selection Criterion**: For each patch, retain cell with max(`e^l × e^w`)
- **Result**: Downsampled to `10×10×7` state (100 tokens)
- **Purpose**: Reduce action space from 10,000 to 100 positions

### Code Implementation
**Match**: ✅ Correctly implemented

```python
# models/encoders.py:120-144
def _downsample(self, container_state):
    # Step 2: Compute e_i * e_j for each element in each patch
    products = patches[..., 1] * patches[..., 2]  # (B, nph, npw, ph, pw)

    # Step 3: Find the index of the max in each patch
    products_flat = products.reshape(B, num_patches_h, num_patches_w, -1)
    flat_idx = torch.argmax(products_flat, dim=-1)

    # Step 4: Convert to global indices
    i_global = i_base * self.patch_size_h + i_patch
    j_global = j_base * self.patch_size_w + j_patch

    # Step 5: Gather features efficiently
    features_reshaped = container_state[batch_idx, i_global, j_global, :]
```

**Notes**:
- Uses features at indices 1 and 2 (right × down distances)
- Returns both downsampled features and global (i,j) coordinates for position mapping
- Patch size: 10×10 as specified in paper

---

## 4. Network Architecture

### Paper Specifications

| Component | d_model | n_heads | num_layers |
|-----------|---------|---------|------------|
| Box Encoder | 128 | 4 | 2 |
| Container Encoder | 128 | 4 | 2 |
| Position Decoder | 128 | 8 | 2 |
| Selection Decoder | 128 | 8 | 2 |
| Orientation Decoder | 128 | 8 | 2 |
| Value Network | 128 | 8 | 1 (decoder) |

### Code Implementation

```python
# models/policy.py:9-18 (defaults)
def __init__(self,
             box_d_model=128, box_n_head=4, box_num_encoder_layers=2,
             cont_d_model=128, cont_n_head=4, cont_num_encoder_layers=2,
             pos_d_model=128, pos_n_head=8, pos_num_decoder_layers=2,
             sel_d_model=128, sel_n_head=8, sel_num_decoder_layers=2,
             orient_d_model=128, orient_n_head=8, orient_num_decoder_layers=2,
             dim_feedforward=512, dropout=0.1)
```

**Match**: ✅ All architectural parameters match paper specifications

**Minor Difference**:
- Paper doesn't explicitly mention `dim_feedforward=512` and `dropout=0.1`, but these are standard Transformer defaults

---

## 5. Reward Function

### Paper Description
```
reward = g_{i-1} - g_i
g_i = W×L×H̃_i - Σ(w_j × l_j × h_j)
```
Where:
- `H̃_i`: Stacked height at step i
- `g_i`: Gap between container volume and packed box volume

### Code Implementation

```python
# envs/packingEnv.py:166-170
def _calculate_g(self):
    if not self.packed_boxes_info: return 0.0
    max_h_stack = np.max(self.container_height_map) if self.container_height_map.any() else 0.0
    total_volume_packed_boxes = sum(b['oriented_dims'][0] * b['oriented_dims'][1] * b['oriented_dims'][2] for b in self.packed_boxes_info)
    return (self.container_L * self.container_W * max_h_stack) - total_volume_packed_boxes
```

```python
# envs/packingEnv.py:229-231
g_prev = self.current_g
self.current_g = self._calculate_g()
reward += (g_prev - self.current_g)  # Positive if g decreased
```

**Match**: ✅ Core formula matches

**Additional Implementation Details Not in Paper**:
- **Step penalty**: `-0.01` per action (line 209)
- **Invalid placement penalty**: `-10^7` (line 235)
- These penalties are mentioned as standard practice but not explicitly in paper's reward section

---

## 6. Training Hyperparameters

### Paper Specifications
| Parameter | Paper Value |
|-----------|-------------|
| Container size (L×W) | 100×100 |
| Box dimensions (l,w,h) | `[L/10, L/2], [W/10, W/2], [min(L,W)/10, max(L,W)/2]` |
| Policy learning rate | 1e-5 |
| Value learning rate | 1e-4 |
| Optimizer | Adam |
| PPO γ (gamma) | 0.99 |
| GAE λ (lambda) | 0.96 |
| PPO ε (epsilon/clip) | 0.12 |
| Training duration | 4 days to 1 week |

### Code Implementation

```python
# train.py:118-119
policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-5)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-4)
```

```python
# train.py:60, 114
def compute_returns_and_advantages(trajectory, gamma=0.99, lamb=0.96, device=DEVICE_STR):
def train_ppo(..., beta=0.01):
```

```python
# train.py:82
def compute_ppo_loss(..., epsilon=0.12, beta=0.01, ...):
```

**Match**: ✅ All hyperparameters match

**Differences**:
1. **PPO epochs**: Code default is 17 (line 168), but paper states 27 should be used for optimal results
2. **Beta (entropy coefficient)**: Code default is 0.01, paper mentions using β but doesn't specify exact value
3. **Default training epochs**: Code sets default to 1,000,000 (line 167)

---

## 7. PPO Loss Function

### Paper Formula
```
L_actor = -min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)
L_critic = MSE(V_θc(s_t), Â_t + V_θc(s_t))
L_entropy = -Σ(π_θs log π_θs + π_θo log π_θo + π_θp log π_θp)
L = L_actor + L_critic + β×L_entropy
```

### Code Implementation

```python
# train.py:82-111
def compute_ppo_loss(policy_net, value_net, container_states, unpacked_boxes_states,
                     old_probs, advantages, returns, epsilon=0.12, beta=0.01, device=DEVICE_STR):
    # Compute new probabilities
    new_probs_tensor = torch.stack([torch.stack([lp for lp in lps]) if isinstance(lps, (list, tuple)) else lps for lps in new_probs])
    ratio = new_probs_tensor.prod(dim=1) / old_probs

    # Actor loss (PPO clipping)
    surr1 = ratio*advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # Critic loss (MSE)
    values = torch.stack([value_net(...) for ...]).squeeze().to(device)
    critic_loss = F.mse_loss(values, returns)

    # Entropy loss
    probs_tensor = torch.stack([torch.tensor(t, dtype=torch.float32) for t in new_probs]).to(device)
    log_probs_tensor = torch.stack([torch.tensor(t, dtype=torch.float32) for t in new_log_probs]).to(device)
    entropy_loss = -torch.sum(probs_tensor * log_probs_tensor)

    total_loss = actor_loss + critic_loss + beta * entropy_loss
    return actor_loss, critic_loss, beta*entropy_loss, total_loss
```

**Match**: ✅ Formula correctly implemented

**Notes**:
- Critic loss uses MSE between values and returns (advantage + old_value)
- Entropy computed as sum over all three action distributions
- Ratio computed as product of three sub-action probability ratios

---

## 8. Value Network Architecture

### Paper Description
- Uses same encoders as policy network
- Transformer decoder with learned query token
- Outputs scalar state value V(s_t)
- **Separate parameters** from policy network (not shared)

### Code Implementation

```python
# models/value.py:48-49
# Learned query token
self.value_token = nn.Parameter(torch.zeros(1, 1, dec_d_model))
```

```python
# models/value.py:110-118
# Expand query token for batch
query = self.value_token.expand(B, -1, -1)  # (B, 1, d)

# Cross-attention over concatenated encodings
memory = torch.cat([box_tokens, container_tokens], dim=1)
dec_output = self.decoder(query, memory)

# MLP head to scalar
value = self.mlp_head(dec_output.squeeze(1)).squeeze(-1)
```

**Match**: ✅ Correctly implemented

**Key Details**:
- Value network has its own separate encoder instances (lines 25-45)
- Parameters are NOT shared with policy network
- Uses decoder with 1 layer by default (can be configured)

---

## 9. Box Orientation Handling

### Paper Description
- 6 orthogonal orientations: `(l,w,h), (l,h,w), (w,l,h), (w,h,l), (h,l,w), (h,w,l)`
- Box encoder averages embeddings to be rotation-invariant
- Orientation decoder generates embeddings for all 6 orientations of selected box

### Code Implementation

```python
# models/decoders.py:99-113
def get_box_orientations(box_dims_tensor):
    l, w, h = box_dims_tensor[0], box_dims_tensor[1], box_dims_tensor[2]
    orientations = [
        [l, w, h], [l, h, w], [w, l, h],
        [w, h, l], [h, l, w], [h, w, l]
    ]
    return torch.tensor(orientations, dtype=box_dims_tensor.dtype, device=box_dims_tensor.device)
```

```python
# models/encoders.py:20-26 (Box encoder averaging for rotation invariance)
l_embed = self.embed_l(unpacked_box_state[..., 0].unsqueeze(-1))
w_embed = self.embed_w(unpacked_box_state[..., 1].unsqueeze(-1))
h_embed = self.embed_h(unpacked_box_state[..., 2].unsqueeze(-1))
stacked_embeddings = torch.stack([l_embed, w_embed, h_embed], dim=2)
avg_embeddings = torch.mean(stacked_embeddings, dim=2)
```

**Match**: ✅ Correctly implemented

---

## 10. Testing Procedure

### Paper Description
- Tested on N=20, 30, 50 boxes
- 1024 randomly generated test instances
- **Sample 16 solutions per instance** (vs 128 in prior work)
- Report best solution among samples
- Metrics: Utilization Rate (UR) and computation time

### Code Implementation

```python
# test.py:13-31
def test_agent(env, policy_net, max_steps=500, device=DEVICE_STR):
    obs, info = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        container_state_T = torch.tensor(obs['container_state'], dtype=torch.float32).unsqueeze(0).to(device)
        unpacked_boxes_state_T = torch.tensor(env.unpacked_boxes, dtype=torch.float32).unsqueeze(0).to(device)
        action, _ , __annotations__ = policy_net(container_state_T, unpacked_boxes_state_T, deterministic_selection=False)
        # Execute action...
```

**Partial Match**: ⚠️

**Differences**:
- Code provides single test run (not 16 samples per instance as in paper)
- User must manually run multiple times to replicate paper's sampling strategy
- No batch testing script for 1024 instances provided
- Uses `deterministic_selection=False` (sampling), paper uses sampling for test too

---

## 11. Key Differences Summary

| Aspect | Paper | Code | Impact |
|--------|-------|------|--------|
| PPO sub-epochs | Not clearly stated, but 27 used | Default: 17, configurable via `--ppo_epochs` | Medium - affects convergence |
| Test sampling | 16 samples per instance | Single run (manual repetition needed) | High - affects reported metrics |
| Invalid placement penalty | Not specified | -10^7 | Low - reasonable default |
| Step penalty | Not specified | -0.01 | Low - standard practice |
| Max episode steps | Not specified | 3× number of boxes | Low - prevents infinite loops |
| Beta (entropy coef) | Mentioned but no value | 0.01 (configurable) | Low - standard value |

---

## 12. Novel Contributions Implementation Status

All three major contributions from the paper are **correctly implemented**:

1. ✅ **Novel State Representation**: Separate container state and box state (no packing history)
   - Eliminates multiple states representing same container configuration
   - Implemented in `envs/packingEnv.py` and used throughout

2. ✅ **Plane Features**: Directional distance features for container
   - `{h, e^l, e^w, e^-l, e^-w, f^l, f^w}` fully implemented
   - Ablation study in paper showed 4.5% improvement for 50 boxes

3. ✅ **Action Order**: Position → Selection → Orientation
   - Previous work used: Selection → Position → Orientation
   - Paper's rationale: More intuitive for large box counts
   - Ablation study showed 5% improvement for 50 boxes

---

## 13. Code Quality Observations

### Strengths
1. Clean separation of concerns (envs, models, training, testing)
2. Configurable via command-line arguments and YAML
3. GPU/MPS/CPU device handling
4. Comprehensive environment with rendering capabilities (GIF, PNG output)

### Areas for Improvement
1. **Missing batch testing script**: Paper tests on 1024 instances with 16 samples each
2. **No checkpointing during training**: Training takes 4-7 days
3. **No early stopping**: Trains for fixed number of epochs
4. **Limited documentation**: Comments could explain "why" not just "what"
5. **Hardcoded container size in encoder**: `original_dim_h=100, original_dim_w=100`

---

## 14. Reproducibility Assessment

### Can Reproduce Paper Results: ⚠️ Partially

**What's Reproducible:**
- ✅ Network architecture matches exactly
- ✅ Training hyperparameters match
- ✅ State representation and features match
- ✅ PPO algorithm matches

**What's Missing for Full Reproduction:**
- ❌ Batch testing script (1024 instances × 16 samples)
- ❌ Results aggregation and statistical analysis
- ❌ Comparison with baseline methods
- ❌ Seed management for reproducibility

**Recommended Additions:**
```python
# Missing: batch_test.py
def batch_test(policy_path, num_instances=1024, samples_per_instance=16, num_boxes=50):
    results = []
    for i in range(num_instances):
        best_ur = 0
        for sample in range(samples_per_instance):
            ur = test_single_instance(policy_path, num_boxes)
            best_ur = max(best_ur, ur)
        results.append(best_ur)
    return np.mean(results), np.std(results)
```

---

## Conclusion

The implementation is **highly faithful** to the paper's methodology. The core innovations (state representation, plane features, action ordering) are correctly implemented. The main differences are:

1. **Minor**: Default PPO epochs (17 vs likely 27 optimal)
2. **Moderate**: Missing convenience scripts for batch testing
3. **Low impact**: Small details like step penalties not explicitly mentioned in paper

Overall Assessment: **85/100** for paper-code alignment
- Deductions: Batch testing (-10), documentation (-5)