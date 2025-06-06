import torch
import torch.nn as nn
from torch.distributions import Categorical


class PositionDecoder(nn.Module):
    def __init__(self, d_model=128, n_head=8, num_decoder_layers=2,
                 dim_feedforward=512, dropout=0.1,
                 container_feature_dim=7): # Num features in original s_c_ds_flat cells
        super(PositionDecoder, self).__init__()
        self.d_model = d_model

        # Core Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # FC layer after Transformer Decoder (leading to Softmax)
        self.fc_for_action_probs = nn.Linear(d_model, 1) # Outputs one logit per item in container_encoding sequence
        self.container_feature_dim = container_feature_dim
        # Embedding layer for the feature extracted from the original Container State
        self.embedding_container_state_feature = nn.Linear(container_feature_dim, d_model)
        

        self.fc_final_pos_embedding = nn.Linear(d_model, d_model)


    def forward(self, container_encoding, box_encoding, container_state, ij_global,
                deterministic_selection=False):
        """
        Follows the flow from the diagram (Fig 4a / user image) to produce a
        chosen position action and its corresponding position embedding.

        Args:
            container_encoding (Tensor): Shape (batch_size, num_patches_total, d_model). Query.
            box_encoding (Tensor): Shape (batch_size, num_unpacked_boxes, d_model). Memory.
            s_c_ds_flat (Tensor): Downsampled original container state features.
                                  Shape (batch_size, num_patches_total, container_feature_dim).
            deterministic_selection (bool): If True, use argmax. Else, sample from distribution.

        Returns:
            chosen_pos_idx (Tensor): The chosen position index (flat). Shape (batch_size,).
            log_prob_chosen_pos (Tensor): Log probability of the chosen_pos_idx. Shape (batch_size,).
            position_embedding (Tensor): Embedding for the chosen_pos_idx. Shape (batch_size, d_model).
            position_probs_all (Tensor): Full probability distribution over all positions.
                                         Shape (batch_size, num_patches_total). (For entropy in PPO).
        """
        batch_size = container_encoding.shape[0]

        # 1. Transformer Decoder
        # Input: container_encoding (q), box_encoding (k,v)
        decoder_output = self.transformer_decoder(tgt=container_encoding, memory=box_encoding)
        # decoder_output shape: (batch_size, num_patches_total, d_model)

        # 2. FC Layer
        fc_output = self.fc_for_action_probs(decoder_output).squeeze(-1) # (batch_size, num_patches_total)

        # 3. Softmax (to get probability distribution)
        position_probs_all = torch.softmax(fc_output, dim=-1)

        # 4. Sample/Argmax (Determine the "Position Action")
        if deterministic_selection:
            chosen_pos_idx = torch.argmax(position_probs_all, dim=1)
        else: # Sample
            pos_distribution = Categorical(probs=position_probs_all)
            chosen_pos_idx = pos_distribution.sample()
        
        # Calculate log probability of the chosen action (needed for training)
        # Using clamped probabilities for numerical stability if creating a new distribution object
        clamped_probs = torch.clamp(position_probs_all, min=1e-9)
        dist_for_log_prob = Categorical(probs=clamped_probs)
        log_prob_chosen_pos = dist_for_log_prob.log_prob(chosen_pos_idx)

        # 5. Compute Position Embedding based on the chosen_pos_idx
        # 5a. Extract from Container Encoding
        # chosen_pos_idx (B,) -> (B,1,1) -> (B,1,d_model) for gather
        feature_from_container_encoding = container_encoding[:, chosen_pos_idx, :]  # (batch_size, d_model)
        # shape: (batch_size, d_model)
        feature_from_container_state = torch.zeros(batch_size, self.container_feature_dim, device=container_state.device)
        for batch in range(batch_size): 
            feature_from_container_state[batch] = container_state[batch, ij_global[batch][chosen_pos_idx][0], ij_global[batch][chosen_pos_idx][1]]  # (batch_size, container_feature_dim)
        # shape: (batch_size, container_feature_dim)
        embedded_feature_from_container_state = self.embedding_container_state_feature(feature_from_container_state)
        # shape: (batch_size, d_model)

        # 5c. Add them
        added_features = feature_from_container_encoding + embedded_feature_from_container_state
        # shape: (batch_size, d_model)

        # 5d. Final FC layer for Position Embedding (as per paper text supplementing Fig 4a)
        position_embedding = self.fc_final_pos_embedding(added_features)
        # shape: (batch_size, d_model)

        return chosen_pos_idx, log_prob_chosen_pos, position_embedding, position_probs_all
    

def get_box_orientations(box_dims_tensor):
    """
    Generates 6 orthogonal orientations for a given box_dims (l, w, h).
    Args:
        box_dims_tensor (Tensor): Tensor of shape (3,) representing (length, width, height).
    Returns:
        Tensor: Tensor of shape (6, 3) representing 6 orientations.
    """
    l, w, h = box_dims_tensor[0], box_dims_tensor[1], box_dims_tensor[2]
    orientations = [
        [l, w, h], [l, h, w], [w, l, h],
        [w, h, l], [h, l, w], [h, w, l]
    ]
    # Ensure output is a tensor on the same device and dtype
    return torch.tensor(orientations, dtype=box_dims_tensor.dtype, device=box_dims_tensor.device)

class SelectionDecoder(nn.Module): 
    def __init__(self, d_model=128, n_head=8, num_decoder_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super(SelectionDecoder, self).__init__()
        self.d_model = d_model
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_for_action_probs = nn.Linear(d_model, 1)
        self.embed_dim_l_for_orientation = nn.Linear(1, d_model)
        self.embed_dim_w_for_orientation = nn.Linear(1, d_model)
        self.embed_dim_h_for_orientation = nn.Linear(1, d_model)

    def forward(self, box_encoding, position_embedding, unpacked_box_state,
                forced_action_idx=None, deterministic_selection=False):
        batch_size = box_encoding.shape[0]
        #memory_position_embedding = position_embedding.unsqueeze(1)
        memory_position_embedding = position_embedding

        decoder_output = self.transformer_decoder(tgt=box_encoding, memory=memory_position_embedding)
        selection_logits = self.fc_for_action_probs(decoder_output).squeeze(-1)
        box_selection_probs_all = torch.softmax(selection_logits, dim=-1)

        if forced_action_idx is not None:
            chosen_box_idx = forced_action_idx.long()
        elif deterministic_selection:
            chosen_box_idx = torch.argmax(box_selection_probs_all, dim=1)
        else:
            box_dist = Categorical(probs=box_selection_probs_all)
            chosen_box_idx = box_dist.sample()

        clamped_probs_box = torch.clamp(box_selection_probs_all, min=1e-9)
        dist_for_log_prob_box = Categorical(probs=clamped_probs_box)
        log_prob_chosen_box = dist_for_log_prob_box.log_prob(chosen_box_idx)

        idx_for_gather_dims = chosen_box_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, unpacked_box_state.shape[-1])
        selected_box_dims_batch = torch.gather(unpacked_box_state, 1, idx_for_gather_dims).squeeze(1)
        
        all_orientations_list = [get_box_orientations(sbd) for sbd in selected_box_dims_batch]
        all_orientations_batch = torch.stack(all_orientations_list, dim=0)
        
        l_oriented = all_orientations_batch[..., 0].unsqueeze(-1)
        w_oriented = all_orientations_batch[..., 1].unsqueeze(-1)
        h_oriented = all_orientations_batch[..., 2].unsqueeze(-1)
        l_embeds_oriented = self.embed_dim_l_for_orientation(l_oriented)
        w_embeds_oriented = self.embed_dim_w_for_orientation(w_oriented)
        h_embeds_oriented = self.embed_dim_h_for_orientation(h_oriented)
        stacked_orientation_embeds = torch.stack([l_embeds_oriented, w_embeds_oriented, h_embeds_oriented], dim=2)
        box_orientation_embedding = torch.mean(stacked_orientation_embeds, dim=2)
        return chosen_box_idx, log_prob_chosen_box, box_orientation_embedding, box_selection_probs_all

class OrientationDecoder(nn.Module): 
    def __init__(self, d_model=128, n_head=8, num_decoder_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super(OrientationDecoder, self).__init__()
        self.d_model = d_model
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_for_action_probs = nn.Linear(d_model, 1)

    def forward(self, box_orientation_embedding, position_embedding,
                forced_action_idx=None, deterministic_selection=False):
        #memory_position_embedding = position_embedding.unsqueeze(1)
        memory_position_embedding = position_embedding
        decoder_output = self.transformer_decoder(tgt=box_orientation_embedding, memory=memory_position_embedding)
        orientation_logits = self.fc_for_action_probs(decoder_output).squeeze(-1)
        orientation_probs_all = torch.softmax(orientation_logits, dim=-1)

        if forced_action_idx is not None:
            chosen_orient_idx = forced_action_idx.long()
        elif deterministic_selection:
            chosen_orient_idx = torch.argmax(orientation_probs_all, dim=1)
        else:
            orient_dist = Categorical(probs=orientation_probs_all)
            chosen_orient_idx = orient_dist.sample()
            
        clamped_probs_orient = torch.clamp(orientation_probs_all, min=1e-9)
        dist_for_log_prob_orient = Categorical(probs=clamped_probs_orient)
        log_prob_chosen_orient = dist_for_log_prob_orient.log_prob(chosen_orient_idx)
        return chosen_orient_idx, log_prob_chosen_orient, orientation_probs_all
