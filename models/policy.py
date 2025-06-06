import torch
import torch.nn as nn
import math
from models.encoders import *
from models.decoders import *
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self,
                 box_d_model=128, box_n_head=4, box_num_encoder_layers=2,
                 cont_original_dim_h=100, cont_original_dim_w=100,
                 cont_patch_size_h=10, cont_patch_size_w=10,
                 cont_feature_dim=7, cont_d_model=128, cont_n_head=4,
                 cont_num_encoder_layers=2,
                 pos_d_model=128, pos_n_head=8, pos_num_decoder_layers=2,
                 sel_d_model=128, sel_n_head=8, sel_num_decoder_layers=2,
                 orient_d_model=128, orient_n_head=8, orient_num_decoder_layers=2, 
                 dim_feedforward=512, dropout=0.1):
        super(PolicyNetwork, self).__init__()

        # --- Instantiate Encoders ---
        self.box_encoder = BoxEncoder(
            d_model=box_d_model, n_head=box_n_head,
            num_encoder_layers=box_num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        
        self.container_encoder = ContainerEncoder(
            original_dim_h=cont_original_dim_h, original_dim_w=cont_original_dim_w,
            patch_size_h=cont_patch_size_h, patch_size_w=cont_patch_size_w,
            feature_dim=cont_feature_dim, d_model=cont_d_model, n_head=cont_n_head,
            num_encoder_layers=cont_num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )

        # --- Instantiate Decoders ---
        self.position_decoder = PositionDecoder(
            d_model=pos_d_model, n_head=pos_n_head,
            num_decoder_layers=pos_num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            container_feature_dim=cont_feature_dim 
        )
        self.selection_decoder = SelectionDecoder(
            d_model=sel_d_model, n_head=sel_n_head,
            num_decoder_layers=sel_num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.orientation_decoder = OrientationDecoder( 
            d_model=orient_d_model, n_head=orient_n_head,
            num_decoder_layers=orient_num_decoder_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout
        )

    def act(self, original_container_state, unpacked_box_state, deterministic_selection=False):
        """
        Generates actions for environment interaction (rollout or testing).
        Determines chosen actions and their log probabilities.

        Args:
            original_container_state (Tensor): Current container state.
            unpacked_box_state (Tensor): Current unpacked boxes state.
            deterministic_selection (bool): True for argmax (testing), False for sampling (training rollout).

        Returns:
            chosen_pos_idx (Tensor)
            chosen_box_idx (Tensor)
            chosen_orient_idx (Tensor)
            log_prob_pos (Tensor)
            log_prob_box (Tensor)
            log_prob_orient (Tensor)
        """
        # 1. Encode inputs
        box_encoding = self.box_encoder(unpacked_box_state)
        #container_encoding, s_c_ds_flat = self.container_encoder(original_container_state)
        container_encoding, downsampled_state, ij_global = self.container_encoder(original_container_state)

        # 2. Position Decoding
        chosen_pos_idx, log_prob_chosen_pos, position_embedding, position_probs_all = self.position_decoder(
            container_encoding=container_encoding,
            box_encoding=box_encoding, 
            container_state=container_encoding,
            ij_global=ij_global,
            deterministic_selection=deterministic_selection
        )

        # 3. Selection Decoding
        chosen_box_idx, log_prob_chosen_box, box_orientation_embedding, box_selection_probs_all = self.selection_decoder(
            box_encoding=box_encoding,
            position_embedding=position_embedding,
            unpacked_box_state=unpacked_box_state,  # Assuming boxes is a list of tuples
            deterministic_selection=deterministic_selection  # Set to True for greedy selection
        )


        chosen_orient_idx, log_prob_chosen_orient, orientation_probs_all = self.orientation_decoder(
            box_orientation_embedding=box_orientation_embedding,
            position_embedding=position_embedding,
            deterministic_selection=deterministic_selection  # Set to True for greedy selection
        )

        return (ij_global[0][chosen_pos_idx], chosen_box_idx, chosen_orient_idx), \
               (log_prob_chosen_pos, log_prob_chosen_box, log_prob_chosen_orient), \
                (position_probs_all, box_selection_probs_all, orientation_probs_all)


    def evaluate_actions(self, log_prob_chosen_pos, log_prob_chosen_box, log_prob_chosen_orient, position_probs_all, box_selection_probs_all, orientation_probs_all):
        """
        Evaluates stored actions under the current policy for PPO updates.
        Computes log probabilities of these specific actions and the entropy of action distributions.

        Args:
            original_container_state (Tensor): Stored container state.
            unpacked_box_state (Tensor): Stored unpacked boxes state.
            action_pos (Tensor): Stored position action.
            action_box (Tensor): Stored box selection action.
            action_orient (Tensor): Stored orientation action.

        Returns:
            total_log_probs (Tensor): Sum of log_probs for the provided action sequence.
            total_entropy (Tensor): Sum of entropies for the action distributions.
        """
        entropy_pos = Categorical(probs=position_probs_all).entropy()

        entropy_box = Categorical(probs=box_selection_probs_all).entropy()

        entropy_orient = Categorical(probs=orientation_probs_all).entropy()

        total_log_probs = log_prob_chosen_pos + log_prob_chosen_box + log_prob_chosen_orient
        total_entropy = entropy_pos + entropy_box + entropy_orient

        return total_log_probs, total_entropy