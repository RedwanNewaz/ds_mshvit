import torch
import torch.nn as nn
import torch.nn.functional as F

TOKENIZERS = ["Patchify", "Sinkhorn", "SharedPatchify", "SharedSinkhorn"]


# Sinkhorn implementation from https://github.com/magicleap/SuperGluePretrainedNetwork
def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, eps, iters):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    norm = - (ms + ns).log() #-M+N in log space
    log_mu = norm.expand(m) # 1xM vector with value -log(M+N)
    log_nu = norm.expand(n) # 1xN vector with value -log(M+N)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1) # add batch dimension

    # scores is the expoenential 
    Z = log_sinkhorn_iterations(scores/eps, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    Z = Z.exp()
    return Z


class Patchify(torch.nn.Module):
    """
    Inspired by Ross Weirghtman's implementation https://github.com/rwightman/pytorch-image-models/blob/dc422820eca4e550a4057561e595fc8b36209137/timm/models/layers/patch_embed.py#L15
    """
    def __init__(self, img_size, patch_size, patch_dim, token_dim, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_tokens = self.grid_size**2

        print(self.img_size, self.patch_size, self.grid_size, self.num_tokens)

        self.proj = nn.Conv2d(patch_dim, token_dim, (patch_size, patch_size), (patch_size, patch_size))

    def forward(self, x, prev_x = None):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        x = self.proj(x) # B x D x H x W -> B x D' x H' x W'
        x = x.flatten(2,3) #  B x D' x H' x W'  ->  B x D' x H'W'
        x_proj = x.transpose(1,2) #  B x D' x H'W' ->  B x H'W' x D' 

        if prev_x is not None:
            x = torch.cat((x_proj, prev_x), dim=1)
        else:
            x = x_proj

        return x, x_proj

class Sinkhorn(torch.nn.Module):
    def __init__(self, patch_dim, token_dim, num_clusters, eps, iters, l2_normalize, **kwargs):
        super().__init__()
        self.v = nn.Parameter(torch.randn(num_clusters,token_dim)) # Cluster centers, drawn from zero mean unit variance Gaussian
        self.proj = nn.Conv2d(patch_dim, token_dim, 1, 1)
        self.num_tokens = num_clusters # Amount of clusters
        self.eps = eps 
        self.iters = iters 
        self.l2_normalize = l2_normalize

    def forward(self, x, prev_x = None):
        x = self.proj(x) # B x D x H x W -> B x D' x H x W
        x = x.flatten(2,3) #  B x D' x H' x W'  ->  B x D' x HW
        x_proj = x.transpose(1,2) #  B x D' x HW ->  B x HW x D' 

        b, hw, d = x_proj.shape
        if prev_x is not None:
            x = torch.cat((x_proj, prev_x), dim=1)
        else:
            x = x_proj

        if self.l2_normalize:
            x = F.normalize(x, p = 2, dim = -1)
        
        ## Ensure clusters are always unit vectors, and require no gradients
        with torch.no_grad():
            w = self.v.clone()
            if self.l2_normalize:
                w = F.normalize(w, p=2, dim = -1)
            self.v.copy_(w)
        clusters  = self.v[None].expand(b, -1, d)# B x Cluster x D'
        
        scores = torch.bmm(x, clusters.transpose(1,2)) # B x HW x D' and B  x D' x Cluster  -> B x HW x Cluster

        weights = log_optimal_transport(scores.transpose(1,2), self.eps, self.iters).transpose(1,2)

        # Batch matrix multiply approach:  B x D x HW and B x HW x Clusters  =  B x D x Cluseters   -> B x Clusters x D
        v_tilde = torch.bmm(x.transpose(1,2), weights).transpose(1,2) # Per input example calculate the weighted sum across the features for each cluster k        

        return v_tilde, x_proj

class SharedPatchify(torch.nn.Module):
    """
    Inspired by Ross Weirghtman's implementation https://github.com/rwightman/pytorch-image-models/blob/dc422820eca4e550a4057561e595fc8b36209137/timm/models/layers/patch_embed.py#L15
    """
    def __init__(self, img_size, patch_size, patch_dim, token_dim, keys, **kwargs):
        super().__init__()
        self.img_size = {key: img_size[idx] for idx, key in enumerate(keys)}
        self.patch_size = patch_size
        self.grid_size = [img_size[idx] // patch_size[idx] for idx in range(len(self.patch_size))]
        self.num_tokens = {key: self.grid_size[idx]**2 for idx, key in enumerate(keys)}

        self.proj = nn.ModuleDict({key: nn.Conv2d(patch_dim[idx], token_dim, (patch_size[idx], patch_size[idx]), (patch_size[idx], patch_size[idx])) for idx, key in enumerate(keys)})

    def forward(self, x, prev_x = None, key=""):
        B, C, H, W = x.shape
        assert H == self.img_size[key] and W == self.img_size[key], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[key]}*{self.img_size[key]})."
        
        x = self.proj[key](x) # B x D x H x W -> B x D' x H x W
        x = x.flatten(2,3) #  B x D' x H' x W'  ->  B x D' x HW
        x_proj = x.transpose(1,2) #  B x D' x HW ->  B x HW x D' 

        b, hw, d = x_proj.shape
        if prev_x is not None:
            x = torch.cat((x_proj, prev_x), dim=1)
        else:
            x = x_proj

        return x, x_proj

class SharedSinkhorn(torch.nn.Module):
    def __init__(self, patch_dim, token_dim, num_clusters, eps, iters, l2_normalize, keys, **kwargs):
        super().__init__()
        self.v = nn.Parameter(torch.randn(num_clusters,token_dim)) # Cluster centers, drawn from zero mean unit variance Gaussian
        self.proj = nn.ModuleDict({key: nn.Conv2d(patch_dim[idx], token_dim, 1, 1) for idx, key in enumerate(keys)})
        self.num_tokens = num_clusters # Amount of clusters
        self.num_tokens = {key: num_clusters for key in keys}
        self.eps = eps 
        self.iters = iters 
        self.l2_normalize = l2_normalize

    def forward(self, x, prev_x=None, key=""):
    
        x = self.proj[key](x) # B x D x H x W -> B x D' x H x W
        x = x.flatten(2,3) #  B x D' x H' x W'  ->  B x D' x HW
        x_proj = x.transpose(1,2) #  B x D' x HW ->  B x HW x D' 
        
        b, hw, d = x.shape
        
        b, hw, d = x_proj.shape
        if prev_x is not None:
            x = torch.cat((x_proj, prev_x), dim=1)
        else:
            x = x_proj

        if self.l2_normalize:
            x = F.normalize(x, p = 2, dim = -1)
        
        ## Ensure clusters are always unit vectors, and require no 
        with torch.no_grad():
            w = self.v.clone()
            w = F.normalize(w, p=2, dim = -1)
            self.v.copy_(w)
        clusters  = self.v[None].expand(b, -1, d)# B x Cluster x D'
        
        scores = torch.bmm(x, clusters.transpose(1,2)) # B x HW x D' and B  x D' x Cluster  -> B x HW x Cluster

        weights = log_optimal_transport(scores.transpose(1,2), self.eps, self.iters).transpose(1,2)

        # Batch matrix multiply approach:  B x D x HW and B x HW x Clusters  =  B x D x Cluseters   -> B x Clusters x D
        v_tilde = torch.bmm(x.transpose(1,2), weights).transpose(1,2) # Per input example calculate the weighted sum across the features for each cluster k       

        return v_tilde, x_proj