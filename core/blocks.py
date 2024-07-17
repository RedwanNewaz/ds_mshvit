import torch
import torch.nn as nn

from .layers import MHSA, MLP, Fourier


BLOCKS = ["TransformerBlock", "MHSABlock", "FourierBlock", "FNetBlock"]

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={:.2f}'.format(self.drop_prob)

class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_drop=0., attn_drop=0.,
                 block_drop = 0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_mhsa = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_res = DropPath(block_drop) if block_drop > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, proj_drop=proj_drop)

    def forward(self, x):
        x = x + self.drop_res(self.attn_mhsa(self.norm1(x)))
        x = x + self.drop_res(self.mlp(self.norm2(x)))
        return x

class MHSABlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, proj_drop=0., attn_drop=0.,
                  block_drop = 0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_res = DropPath(block_drop) if block_drop > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_res(self.attn(self.norm1(x)))
        return x

class FNetBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., proj_drop=0., block_drop = 0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Fourier()
        self.drop_res = DropPath(block_drop) if block_drop > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, proj_drop=proj_drop)

    def forward(self, x):
        x = x + self.drop_res(self.attn(self.norm1(x)))
        x = x + self.drop_res(self.mlp(self.norm2(x)))
        return x

    
class FourierBlock(nn.Module):

    def __init__(self, dim, block_drop = 0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Fourier()
        self.drop_res = DropPath(block_drop) if block_drop > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_res(self.attn(self.norm1(x)))
        return x
