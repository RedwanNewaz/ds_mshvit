import torch
import torch.nn as nn



class MLP(nn.Module):
    """ 
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    Ross Wightman's implementation: https://github.com/rwightman/pytorch-image-models/blob/dc422820eca4e550a4057561e595fc8b36209137/timm/models/layers/mlp.py#L8
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, proj_drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.proj_drop(x)
        x = self.fc2(x)
        x = self.proj_drop(x)
        return x

class MHSA(nn.Module):
    """
    Multi-Head Self-Attention, without normalization layers.
    
    Ross Wightman's implementation: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L172
    Slight modifications.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # B x N x D

        # Compute queries, keys, and values (using the same linear layer), and then split into the resepctive atention heads
        qkv = self.qkv(x) # B x N x D*3
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # B x N x 3 x N_Heads x D // N_Heads
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 x B x N_Heads x N x D // N_Heads 

        # Separate the queries, keys and values
        q, k, v = qkv[0], qkv[1], qkv[2]   # Each are B x N_Heads x N x D // N_Heads 
        
        # Get similarity scores between queries and keys, and apply softmax + dropout
        attn = torch.matmul(q, k.transpose(-2,-1)) * self.scale  # B x N_Heads x N x D // N_Heads ,  B x N_Heads x D // N_Heads x N   ->  B x N_Heads x N x N
        attn = attn.softmax(dim=-1) # B x N_Heads x N x N
        attn = self.attn_drop(attn)  # B x N_Heads x N x N

        # Apply attention scores over the values, and restructure the tensor back into its original size
        x = torch.matmul(attn, v) # B x N_Heads x N x N ,  B x N_Heads x N x D // N_Heads  ->  B x N_Heads x N x D // N_Heads
        x = x.transpose(1, 2) # B x N_Heads x N x D // N_Heads ->  B x N x N_Heads x D // N_Heads
        x = x.reshape(B, N, C) # B x N x N_Heads x D // N_Heads ->  B x N x D

        # Apply a linear projection
        x = self.proj(x) 
        x = self.proj_drop(x)

        return x

class Fourier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # X = B x N x D
        x = torch.fft.fft(x, dim=-1) ## FFT over the hidden dimension
        x = torch.fft.fft(x, dim=-2).real ## FFT across the token dimension
        return x