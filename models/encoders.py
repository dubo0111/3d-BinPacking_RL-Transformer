import torch
import torch.nn as nn
import math

class BoxEncoder(nn.Module):
    def __init__(self, d_model=128, n_head=4, num_encoder_layers=2, dim_feedforward=512, dropout=0.1):
        super(BoxEncoder, self).__init__()
        self.d_model = d_model
        self.embed_l = nn.Linear(1, d_model)
        self.embed_w = nn.Linear(1, d_model)
        self.embed_h = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu' 
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, unpacked_box_state):
        l_embed = self.embed_l(unpacked_box_state[..., 0].unsqueeze(-1))
        w_embed = self.embed_w(unpacked_box_state[..., 1].unsqueeze(-1))
        h_embed = self.embed_h(unpacked_box_state[..., 2].unsqueeze(-1))
        stacked_embeddings = torch.stack([l_embed, w_embed, h_embed], dim=2)
        avg_embeddings = torch.mean(stacked_embeddings, dim=2)
        box_encoding = self.transformer_encoder(avg_embeddings)
        return box_encoding


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len_h=20, max_len_w=20):
        super(FixedPositionalEncoding, self).__init__()
        self.d_model = d_model

        self.max_len_h = max_len_h
        self.max_len_w = max_len_w

        pe = torch.zeros(max_len_h * max_len_w, d_model)

        self.register_buffer('pe_base', pe) # Placeholder, will be populated in forward


    def forward(self, num_patches_h, num_patches_w):

        pe_h_part = torch.zeros(num_patches_h, self.d_model, device=self.pe_base.device)
        pe_w_part = torch.zeros(num_patches_w, self.d_model, device=self.pe_base.device)

        position_h = torch.arange(0, num_patches_h, dtype=torch.float, device=self.pe_base.device).unsqueeze(1)
        position_w = torch.arange(0, num_patches_w, dtype=torch.float, device=self.pe_base.device).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(self.pe_base.device)

        pe_h_part[:, 0::2] = torch.sin(position_h * div_term)
        if self.d_model % 2 != 0: # Handle odd d_model by not filling the last column for cos
             pe_h_part[:, 1::2] = torch.cos(position_h * div_term)[:,:pe_h_part[:,1::2].size(1)]
        else:
             pe_h_part[:, 1::2] = torch.cos(position_h * div_term)


        pe_w_part[:, 0::2] = torch.sin(position_w * div_term)
        if self.d_model % 2 != 0:
            pe_w_part[:, 1::2] = torch.cos(position_w * div_term)[:,:pe_w_part[:,1::2].size(1)]
        else:
            pe_w_part[:, 1::2] = torch.cos(position_w * div_term)


        combined_pe = pe_h_part.unsqueeze(1) + pe_w_part.unsqueeze(0) # (H, W, D)
        flat_pe = combined_pe.view(-1, self.d_model) # (H*W, D)
        return flat_pe.unsqueeze(0) # (1, H*W, D) for broadcasting


class ContainerEncoder(nn.Module):
    def __init__(self, original_dim_h, original_dim_w, patch_size_h, patch_size_w,
                 feature_dim=7, d_model=128, n_head=4, num_encoder_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super(ContainerEncoder, self).__init__()
        self.original_dim_h = original_dim_h
        self.original_dim_w = original_dim_w
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.feature_dim = feature_dim
        self.d_model = d_model

        assert original_dim_h % patch_size_h == 0, "Original height must be divisible by patch height."
        assert original_dim_w % patch_size_w == 0, "Original width must be divisible by patch width."

        self.num_patches_h = original_dim_h // patch_size_h
        self.num_patches_w = original_dim_w // patch_size_w

        self.embedding = nn.Linear(feature_dim, d_model)
        self.positional_encoding = FixedPositionalEncoding(d_model, self.num_patches_h, self.num_patches_w)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu' 
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

    def _downsample(self, container_state):
        """
        Args:
            container_state: torch.Tensor of shape (B, H, W, F) or (B, H, W, 7)
        Returns:
            features_reshaped: (B, num_patches_h, num_patches_w, 7)
            ij_global: list of (B, num_patches_h * num_patches_w) tuples of (i, j) indices
        """
        B, H, W, F = container_state.shape
        num_patches_h = H // self.patch_size_h
        num_patches_w = W // self.patch_size_w

        # Step 1: Reshape into patches
        patches = container_state.view(
            B, num_patches_h, self.patch_size_h,
            num_patches_w, self.patch_size_w, F
        )  # (B, nph, ph, npw, pw, F)
        patches = patches.permute(0, 1, 3, 2, 4, 5)  # (B, nph, npw, ph, pw, F)

        # Step 2: Compute e_i * e_j for each element in each patch
        products = patches[..., 1] * patches[..., 2]  # (B, nph, npw, ph, pw)

        # Step 3: Find the index of the max in each patch
        products_flat = products.reshape(B, num_patches_h, num_patches_w, -1)  # (B, nph, npw, ph*pw)
        flat_idx = torch.argmax(products_flat, dim=-1)  # (B, nph, npw)
        i_patch = torch.div(flat_idx, self.patch_size_h, rounding_mode='trunc')
        j_patch = flat_idx % self.patch_size_h  # (B, nph, npw)

        # Step 4: Convert to global indices
        i_base = torch.arange(num_patches_h, device=container_state.device).view(1, num_patches_h, 1)
        j_base = torch.arange(num_patches_w, device=container_state.device).view(1, 1, num_patches_w)
        i_global = i_base * self.patch_size_h + i_patch  # (B, nph, npw)
        j_global = j_base * self.patch_size_w + j_patch  # (B, nph, npw)

        # Step 5: Gather features efficiently
        # Prepare batch indices for advanced indexing
        batch_idx = torch.arange(B, device=container_state.device).view(B, 1, 1).expand_as(i_global)
        features_reshaped = container_state[batch_idx, i_global, j_global, :]  # (B, nph, npw, F)

        # Optionally, return ij_global as a list of lists for each batch
        ij_global = [
            list(zip(i_global[b].flatten().tolist(), j_global[b].flatten().tolist()))
            for b in range(B)
        ]
        return features_reshaped, ij_global
    

    def forward(self, container_state):
        downsampled_state_features, ij_global = self._downsample(container_state) # (B, NPH, NPW, F)

        embedded_state = self.embedding(downsampled_state_features) # (B, NPH, NPW, d_model)
        batch_size = embedded_state.shape[0]
        sequence_state = embedded_state.view(batch_size, -1, self.d_model) # (B, NPH*NPW, d_model)
        pos_enc = self.positional_encoding(self.num_patches_h, self.num_patches_w)
        input_with_pos_enc = sequence_state + pos_enc
        container_encoding = self.transformer_encoder(input_with_pos_enc)
        # Return both encoding and the downsampled features for decoder use
        return container_encoding, downsampled_state_features.view(batch_size, -1, self.feature_dim), ij_global

