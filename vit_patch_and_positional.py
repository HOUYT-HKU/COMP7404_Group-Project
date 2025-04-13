
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x shape: (B, 3, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x


class PositionalEmbedding(nn.Module):
    """
    添加可学习的位置编码和 class token
    """
    def __init__(self, n_patches, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

    def forward(self, x):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, C)
        x = x + self.pos_embed
        return x
