import torch
import torch.nn as nn
from models.encoders import BoxEncoder, ContainerEncoder

class ValueNetwork(nn.Module):
    def __init__(self,
        # ---- box-encoder params ----
        box_d_model: int = 128, box_n_head: int = 4,
        box_num_encoder_layers: int = 2,

        # ---- container-encoder params ----
        cont_original_dim_h: int = 100, cont_original_dim_w: int = 100,
        cont_patch_size_h: int = 10,  cont_patch_size_w: int = 10,
        cont_feature_dim: int = 7,    cont_d_model: int = 128,
        cont_n_head: int = 4,         cont_num_encoder_layers: int = 2,

        # ---- decoder & head params ----
        dec_d_model: int = 128, dec_n_head: int = 8, dec_num_layers: int = 1,
        dim_feedforward: int = 512, dropout: float = 0.1,
    ):
        super(ValueNetwork, self).__init__()

        # 1. Box encoder (unpacked boxes → sequence); parameters are NOT shared
        #    with the actor to avoid update interference.
        self.box_encoder = BoxEncoder(
            d_model=box_d_model,
            n_head=box_n_head,
            num_encoder_layers=box_num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # 2. Container encoder (top-view grid → sequence of 10×10 = 100 tokens).
        self.container_encoder = ContainerEncoder(
            original_dim_h=cont_original_dim_h,
            original_dim_w=cont_original_dim_w,
            patch_size_h=cont_patch_size_h,
            patch_size_w=cont_patch_size_w,
            feature_dim=cont_feature_dim,
            d_model=cont_d_model,
            n_head=cont_n_head,
            num_encoder_layers=cont_num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # 3. Learned **query token**.
        #    Shape: (1, 1, d) — duplicated across the batch in forward().
        self.value_token = nn.Parameter(torch.zeros(1, 1, dec_d_model))

        # 4. Single-layer Transformer decoder.
        #    Query = value_token; Key/Value = concat(box_tokens, container_tokens)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_d_model,
            nhead=dec_n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,      # → (B, T, d) everywhere
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=dec_num_layers,
        )

        # 5. Small MLP → scalar.
        self.mlp_head = nn.Sequential(
            nn.Linear(dec_d_model, dec_d_model // 2),
            nn.ReLU(),
            nn.Linear(dec_d_model // 2, 1)      # final scalar value
        )

        # 6. Xavier init for stability (same scheme as original Transformer).
        self._reset_parameters()

    # ------------------------------------------------------------------ #
    # Weight init                                                        #
    # ------------------------------------------------------------------ #
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:  # tensors with >=2 dims are weight matrices
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------ #
    # Forward pass                                                       #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        container_state: torch.Tensor,   # (B, 100, 100, 7)
        box_state: torch.Tensor,         # (B, N_boxes, 3)
    ) -> torch.Tensor:
        """
        Returns
        -------
        value : torch.Tensor, shape (B,)
            Predicted state value V(s_t).
        """

        B = container_state.size(0)  # batch size

        # -- Encode boxes --
        #    Output shape: (B, N_b, d)
        box_tokens = self.box_encoder(box_state)

        # -- Encode container --
        #    container_encoder returns:
        #        • tokens  (B, N_c, d)  for the Transformer, plus
        #        • down-sampled grid & ij_global (ignored here).
        container_tokens, _, _ = self.container_encoder(container_state)

        # -- Concatenate memory tokens --
        #    Combined sequence length = N_b + N_c.
        memory = torch.cat([box_tokens, container_tokens], dim=1)

        # -- Prepare query token for each batch element --
        #    expand() is zero-cost; it returns a view (no new memory).
        query = self.value_token.expand(B, -1, -1)  # (B, 1, d)

        # -- Cross-attention (decoder) --
        #    query attends over memory → (B, 1, d)
        dec_output = self.decoder(query, memory)

        # -- Regression head --
        #    Squeeze the sequence & feature dims → (B,)
        value = self.mlp_head(dec_output.squeeze(1)).squeeze(-1)
        return value
