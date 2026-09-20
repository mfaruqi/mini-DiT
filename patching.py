
import math
import torch
def patchify(x, patch_size):
    """"x: (B, C, H, W) x is a tensor of images of size (C, H, W)
    return: patchified image: (B, N, p*p*C)"""

    B = x.size(0)
    C = x.size(1)
    H = x.size(2)
    W = x.size(3)
    nh = H // patch_size
    nw = W // patch_size
    
    x = x.reshape(B, C, nh, patch_size, nw, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)
    x = x.reshape(B, nh*nw, patch_size*patch_size*C)
    return x

def unpatchify(x, patch_size):
    B, N, patch_dim = x.shape
    C = patch_dim // (patch_size ** 2)
    nh = nw = math.isqrt(N)

    x = x.reshape(B, nh, nw, patch_size, patch_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4)
    x = x.reshape(B, C, nh*patch_size, nw*patch_size)
    return x


def get_2d_sincos_pos_embed(dim, grid_h, grid_w):
    assert dim % 4 == 0
    h = torch.arange(grid_h, dtype=torch.float32)
    w = torch.arange(grid_w, dtype=torch.float32)

    h_grid, w_grid = torch.meshgrid(h, w, indexing='ij')
    h_pos = h_grid.reshape(-1) # becomes (N, ) ex. [0, 0, 1, 1]
    w_pos = w_grid.reshape(-1)

    factor = 10000 ** (torch.arange(dim // 4) / (dim // 4)) # (dim/4,)

    h_angles = h_pos[:, None] / factor # (N, dim/4)
    w_angles = w_pos[:, None] / factor # (N, dim/4)

    h_embed = torch.cat([torch.sin(h_angles), torch.cos(h_angles)], dim=-1)  # (N, dim/2)

    w_embed = torch.cat([torch.sin(w_angles), torch.cos(w_angles)], dim=-1)  # (N, dim/2)

    pos_embed = torch.cat([h_embed, w_embed], dim=-1)  # (N, dim)

    return pos_embed