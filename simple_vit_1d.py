import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        # attn = softmax_one(dots, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(self, packets, subcarries, depth, heads, mlp_dim, dim_head=64):
        super().__init__()

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(subcarries),
            Rearrange("b (p n) s -> b p n s", n=1),
            nn.Conv2d(
                in_channels=packets,
                out_channels=packets,
                kernel_size=(1, 7),
                padding=(0, 3),
                groups=packets,
                bias=False,
            ),
            Rearrange("b p n s -> b (p n) s"),
            nn.LayerNorm(subcarries),
        )

        self.transformer = Transformer(subcarries, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()

    def forward(self, csis):
        x = self.to_patch_embedding(csis)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return x
