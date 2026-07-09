import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TemporalTransformerEncoder(nn.Module):
    def __init__(self,
                 feature_dim=768,      # Swin output dim
                 num_frames=32,
                 num_heads=8,
                 num_layers=4,         # usually 2-6 is enough
                 mlp_ratio=4,
                 num_classes=2,
                 dropout=0.1):
        super().__init__()

        self.num_frames = num_frames
        self.feature_dim = feature_dim

        # Learnable temporal positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, feature_dim))

        # Optional CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x, frame_indices = None):
        # x shape: (B, T, D)   where T = num_frames, D = feature_dim
        B, T, D = x.shape

        if frame_indices is not None:
            # frame_indices shape must be (B, T)
            # We need to gather the correct positional embeddings for each item in the batch.

            # Expand pos_embed to match batch size: (B, max_frames, D)
            expanded_pos_embed = self.pos_embed.expand(B, -1, -1)

            # Expand indices to match feature dimension: (B, T, D)
            # This allows us to use torch.gather to pluck out the correct 768-dim vectors
            gather_indices = frame_indices.unsqueeze(-1).expand(-1, -1, D)

            # Gather the selected embeddings
            selected_pos_embed = torch.gather(expanded_pos_embed, 1, gather_indices)

            x = x + selected_pos_embed
        else:
            x = x + self.pos_embed[:, :T]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)   # (B, T+1, D)

        # Pass through Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Use CLS token for classification
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits

