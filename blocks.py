import torch
from torch import nn
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_proj = nn.Linear(hidden_size, 3*hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)




    def forward(self, x):
        B, N, _ = x.shape # _ is hidden_dim already have it
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim)
        v = v.reshape(B, N, self.num_heads, self.head_dim)

        # change so each head can attend to all tokens
        # before each token had a dimension of 4 heads it was split into
        # now each head has each token hidden_dim / num_heads values of each token
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = q @ k.transpose(-2, -1)

        # sqrt(self.head_dim) because a dot product over head_dim numbers can be large 
        # making softmax unstable
        scores = scores * (self.head_dim ** -0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        attn_out = attn_weights @ v
        attn_out = attn_out.permute(0, 2, 1, 3)
        attn_out = attn_out.reshape(B, N, self.head_dim * self.num_heads)
        out = self.out_proj(attn_out)

        return out
    
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)    
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, x, c):
        """c: (B, hidden_dim)"""
        controls = self.adaLN_modulation(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = \
        controls.chunk(6, dim=-1)

        x_mod = self.norm1(x) * (1 + scale1.unsqueeze(1)) \
                + shift1.unsqueeze(1)
        
        attn_out = self.attn(x_mod)

        x = x + attn_out * gate1.unsqueeze(1)

        x_mod = self.norm2(x) * (1 + scale2.unsqueeze(1)) \
                + shift2.unsqueeze(1)
        
        mlp_out = self.MLP(x_mod)

        x = x + gate2.unsqueeze(1) * mlp_out
        return x
    
    
        
        





        
