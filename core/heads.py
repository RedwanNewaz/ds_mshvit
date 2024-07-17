import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from . import tokenizers as tokenizers

HEADS = ["BaseHead", "MultiScaleViTHead"]

class BaseHead(nn.Module):
    def __init__(self, num_classes, backbone_feature_map_sizes, last_stage, global_pool="avg", sigmoid_loss = True, tresnet_init = False):
        super().__init__()

        self.num_classes = num_classes
        self.feature_map_sizes = backbone_feature_map_sizes
        self.last_stage = last_stage

        if global_pool.lower() == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif global_pool.lower() == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError("The suppleid global pooling layer is not implemented: {}".format(global_pool))
        self.cls_fc = nn.Linear(self.feature_map_sizes[self.last_stage][1], num_classes)

        self.init_weights(sigmoid_loss, tresnet_init)

    def forward(self, x):
        x = self.pool(x[self.last_stage])
        x = torch.flatten(x, 1)
        x = self.cls_fc(x)
        return x

    def init_weights(self, sigmoid_loss = True, tresnet_init = False):
        if tresnet_init:
            self.apply(_tresnet_init_weights)
        else:
            self.apply(_init_weights)

        # Adjust bias at classification layer, if using sigmoid based losses (eg. focal loss, BCEWithLossLogits)
        # See class efficient samples weighting paper 
        if sigmoid_loss:
            head_bias = -math.log(self.num_classes-1)
            nn.init.constant_(self.cls_fc.bias, head_bias)

class MultiScaleViTHead(nn.Module):
    def __init__(self, num_classes, token_dim, representation_size, tokenizer_layer_name, block_type, block_drop, use_pos_embed, pos_embed_drop, 
                 backbone_feature_map_sizes, backbone_feature_maps, transformer_depths, sigmoid_loss, norm_layer, act_layer,
                 tokenizer_kwargs_base, transformer_block_kwargs_base, cross_attention_kwargs, cross_block_type, shared_tower,
                 multiscale_method, late_fusion, cross_scale_all, shared_tokenizer, no_vit_layers, tresnet_init, use_mean_token):
        super().__init__()
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.feature_map_sizes = backbone_feature_map_sizes
        self.shared_tower = shared_tower
        self.shared_tokenizer = shared_tokenizer
        self.backbone_feature_maps = sorted(backbone_feature_maps)
        self.multiscale_method = multiscale_method # SeparateScale, CrossScale-CNN, CrossScale-Token, CrossScale-VIT
        self.use_mean_token = use_mean_token

        if self.use_mean_token:
            self.num_tokens = 0
        else:
            self.num_tokens = 1

        if isinstance(no_vit_layers, (list, type)):
            self.no_vit_layers = sorted(no_vit_layers)
        else:
            self.no_vit_layers = []
        
        if self.shared_tokenizer and "shared" not in tokenizer_layer_name.lower():
            tokenizer_layer_name = "Shared" + tokenizer_layer_name
        self.tokenizer_layer_name = tokenizer_layer_name
        tokenizer_layer = tokenizers.__dict__[tokenizer_layer_name]

        if self.multiscale_method == "SeparateScale":
            self.cross_scale_all = False
            self.combine_cls = True
        else:
            self.cross_scale_all = cross_scale_all
            self.combine_cls = late_fusion
        
        if len(backbone_feature_maps) == 1:
            self.combine_cls = False

        self.vit_layers = sorted(list(set(self.backbone_feature_maps).difference(set(self.no_vit_layers))))
        invalid_no_vit_layer = list(set(self.no_vit_layers).difference(set(self.backbone_feature_maps)))
        assert len(invalid_no_vit_layer) == 0, "Supplied no VIT layers '{}' contain layer not in the backbone feature maps '{}'".format(self.no_vit_layers, self.backbone_feature_maps)

        self.num_tokenizers = len(self.backbone_feature_maps)
        self.num_vits = len(self.vit_layers)

        if self.multiscale_method in ["SeparateScale", "CrossScale-VIT"]:
            assert self.num_tokenizers == self.num_vits, "'No VIT Layers {}' are not allowed for the {} Multi scale method".format(self.no_vit_layers, self.multiscale_method)

        for key_token in tokenizer_kwargs_base.keys():
            val = tokenizer_kwargs_base[key_token]
            if isinstance(val, list):

                if self.shared_tokenizer and key_token != "patch_size": 
                    assert len(val) == 1 
                    tokenizer_kwargs_base[key_token] = val[0]
                else:
                    assert len(val) == self.num_tokenizers or len(val) == 1
                    if len(val) == 1:
                        tokenizer_kwargs_base[key_token] = [val[0]] * self.num_tokenizers
            else:
                if not self.shared_tokenizer:
                    tokenizer_kwargs_base[key_token] = [val] * self.num_tokenizers
                

        for key_trans in transformer_block_kwargs_base.keys():
            val = transformer_block_kwargs_base[key_trans]

            if isinstance(val, list):
                if self.shared_tower: 
                    assert len(val) == 1
                    transformer_block_kwargs_base[key_trans] = val[0]
                else:
                    assert len(val) == self.num_vits or len(val) == 1
                    if len(val) == 1:
                        transformer_block_kwargs_base[key_trans] = [val[0]] * self.num_vits
            else:
                if not self.shared_tower:
                    transformer_block_kwargs_base[key_trans] = [val] * self.num_vits


        if isinstance(transformer_depths, list):
            if self.shared_tower: 
                assert len(transformer_depths) == 1
                transformer_depths = transformer_depths[0]
            else:
                assert len(transformer_depths) == self.num_vits or len(transformer_depths) == 1
                if len(transformer_depths) == 1:
                    transformer_depths = [transformer_depths[0]] * self.num_vits
        else:
            if not self.shared_tower:
                transformer_depths = [transformer_depths] * self.num_vits


        # Setup Tokenizers
        if self.shared_tokenizer:
            t_kw = {**{key_token: tokenizer_kwargs_base[key_token] for key_token in tokenizer_kwargs_base.keys()}, **{"img_size":[self.feature_map_sizes[key][0] for key in self.backbone_feature_maps]}, **{"patch_dim":[self.feature_map_sizes[key][1] for key in backbone_feature_maps]}, **{"keys": [key for key in self.backbone_feature_maps]}}
            self.tokenizer = tokenizer_layer(token_dim=token_dim, **t_kw)
            num_patches = self.tokenizer.num_tokens
            
        else:
            tokenizer_dict = {}
            for idx, key in enumerate(self.backbone_feature_maps):
                t_kw = {**{key_token: tokenizer_kwargs_base[key_token][idx] for key_token in tokenizer_kwargs_base.keys()}, **{"img_size":self.feature_map_sizes[key][0], "patch_dim":self.feature_map_sizes[key][1]}}
                tokenizer_dict[key] = tokenizer_layer(token_dim=token_dim, **t_kw)
            self.tokenizer = nn.ModuleDict(tokenizer_dict)
            num_patches = {key: self.tokenizer[key].num_tokens for key in self.backbone_feature_maps}

        num_patches_cummulative = {}
        num_patches_list = list(num_patches.values())
        num_patches_keys = list(num_patches.keys())
        for idx in range(len(num_patches_list)):
            if len(num_patches_cummulative) == 0:
                num_patches_cummulative[num_patches_keys[idx]] = num_patches_list[idx]
            else:
                if "Patchify" in tokenizer_layer_name:
                    if self.cross_scale_all:
                        num_patches_cummulative[num_patches_keys[idx]] = num_patches_list[idx] + sum(num_patches_cummulative.values())
                    else:
                        num_patches_cummulative[num_patches_keys[idx]] = num_patches_list[idx] + num_patches_cummulative[num_patches_keys[idx-1]]
                else:
                    num_patches_cummulative[num_patches_keys[idx]] = num_patches_list[idx]


        if not self.shared_tower:
            self.transformers = nn.ModuleDict({key: ViT(token_dim=token_dim, representation_size=representation_size, block_type=block_type, block_drop = block_drop,
                                                    depth = transformer_depths[idx], norm_layer=norm_layer, act_layer=act_layer,
                                                    transformer_block_kwargs={key_trans: transformer_block_kwargs_base[key_trans][idx] for key_trans in transformer_block_kwargs_base.keys()}) for idx, key in enumerate(self.vit_layers)})
        else:
            self.transformers = ViT(token_dim=token_dim, representation_size=representation_size, block_type=block_type, block_drop = block_drop,
                                                    depth = transformer_depths, norm_layer=norm_layer, act_layer=act_layer,
                                                    transformer_block_kwargs={key_trans: transformer_block_kwargs_base[key_trans] for key_trans in transformer_block_kwargs_base.keys()})

        # Optional: Change final feature representation size 
        if representation_size:
            self.out_features = representation_size
        else:
            self.out_features = token_dim

        # Optional: Multi-Scale late fusion
        if self.combine_cls:
                self.cross_cls_token = nn.Parameter(torch.zeros(1, 1, self.out_features))
                self.cls_aggregator = cross_block_type(dim = self.out_features, norm_layer = norm_layer, **cross_attention_kwargs)
        else:
            self.cls_aggregator = nn.Identity()


        self.use_pos_embed = use_pos_embed
        if self.use_pos_embed:
            self.pos_drop = nn.Dropout(p=pos_embed_drop)
            
        # Define forward functions + CLS tokens and Positional Embeddings
        if self.multiscale_method == "CrossScale-CNN" or self.multiscale_method == "CrossScale-Token" or self.multiscale_method == "CrossScale-VIT":
            self.forward_func = self.cross_scale_forward

            if not self.use_mean_token:
                self.cls_token = nn.ParameterDict({key: nn.Parameter(torch.zeros(1, 1, token_dim)) for key in self.backbone_feature_maps})
            if self.use_pos_embed:
                self.pos_embed = nn.ParameterDict({key: nn.Parameter(torch.zeros(1, num_patches_cummulative[key] + self.num_tokens, token_dim)) for key in self.backbone_feature_maps})

        elif self.multiscale_method == "SeparateScale":
            self.forward_func = self.separate_scale_forward
            
            if not self.use_mean_token:
                self.cls_token = nn.ParameterDict({key: nn.Parameter(torch.zeros(1, 1, token_dim)) for key in self.vit_layers})
            if self.use_pos_embed:
                self.pos_embed = nn.ParameterDict({key: nn.Parameter(torch.zeros(1, num_patches[key] + self.num_tokens, token_dim)) for key in self.vit_layers})

        # Main Classifier
        self.cls_fc = nn.Linear(self.out_features, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(sigmoid_loss, tresnet_init)

    def init_weights(self, sigmoid_loss = True, tresnet_init = False):
        
        if tresnet_init:
            self.apply(_tresnet_init_weights)
        else:
            self.apply(_init_weights)
        
        # Adjust bias at classification layer, if using sigmoid based losses (eg. focal loss, BCEWithLossLogits)
        # See class efficient samples weighting paper 
        if sigmoid_loss:
            # Adjust bias at classification layer, if using sigmoid based losses (eg. focal loss, BCEWithLossLogits)
            # See class efficient samples weighting paper 
            head_bias = -math.log(self.num_classes-1) if sigmoid_loss else 0. 
            nn.init.constant_(self.cls_fc.bias, head_bias)

        if self.combine_cls:
            torch.nn.init.trunc_normal_(self.cross_cls_token, std=.02)

        for key in self.vit_layers:
            if self.use_pos_embed:
                torch.nn.init.trunc_normal_(self.pos_embed[key], std=.02)
            if not self.use_mean_token:
                torch.nn.init.trunc_normal_(self.cls_token[key], std=.02)
  
    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_set = set()

        if self.use_pos_embed:
            no_weight_set.add("pos_embed")
        if not self.use_mean_token:
            no_weight_set.add("cls_token")
            no_weight_set.add("cross_cls_token")
        return no_weight_set

    def cross_scale_forward(self, x):
        prev_layer = None
        cls_tokens = []
        for idx, key in enumerate(x.keys()):
            # Apply tokenizer
            if self.shared_tokenizer:
                inp_tokens, x_proj = self.tokenizer(x[key], prev_layer, key)
            else:
                inp_tokens, x_proj = self.tokenizer[key](x[key], prev_layer)
            
            # Add CLS token and positon embeddings
            if not self.use_mean_token:
                cls_token = self.cls_token[key].expand(inp_tokens.shape[0], -1, -1)
                tokens = torch.cat((cls_token, inp_tokens), dim=1)
            else:
                tokens = inp_tokens

            if self.use_pos_embed:
                tokens = self.pos_drop(tokens + self.pos_embed[key])

            if key in self.vit_layers:
                # Go through transformer encoder blocks
                if self.shared_tower:
                    cls_token_vit, all_tokens = self.transformers(tokens, self.use_mean_token)
                else:
                    cls_token_vit, all_tokens = self.transformers[key](tokens, self.use_mean_token)
                
                cls_tokens.append(cls_token_vit.unsqueeze(1))
            
            if self.multiscale_method == "CrossScale-CNN":
                layer_tokens = x_proj 
            elif self.multiscale_method == "CrossScale-Token":
                layer_tokens = inp_tokens 
            elif self.multiscale_method == "CrossScale-VIT":
                layer_tokens = all_tokens 

            if self.cross_scale_all:
                if prev_layer is None:
                    prev_layer = layer_tokens
                else:
                    prev_layer = torch.cat((prev_layer, layer_tokens), dim=1)
            else:
                prev_layer = layer_tokens

        return cls_tokens
    
    def separate_scale_forward(self, x):
        cls_tokens = []
        for idx, key in enumerate(x.keys()):
            # Apply tokenizer
            if self.shared_tokenizer:
                inp_tokens, _ = self.tokenizer(x[key], None, key)
            else:
                inp_tokens, _ = self.tokenizer[key](x[key])

            # Add CLS token and positon embeddings
            if not self.use_mean_token:
                cls_token = self.cls_token[key].expand(inp_tokens.shape[0], -1, -1)
                tokens = torch.cat((cls_token, inp_tokens), dim=1)
            else:
                tokens = inp_tokens

            if self.use_pos_embed:
                tokens = self.pos_drop(tokens + self.pos_embed[key])

            # Go through transformer encoder blocks
            if self.shared_tower:
                cls_token_vit, _ = self.transformers(tokens, self.use_mean_token)
            else:
                cls_token_vit, _ = self.transformers[key](tokens, self.use_mean_token)
            
            cls_tokens.append(cls_token_vit.unsqueeze(1))

        return cls_tokens

    def forward(self, x, return_tokens = False, return_weights = True):
        try:
            cls_tokens = self.forward_func(x, return_tokens, return_weights)
        except:
            cls_tokens = self.forward_func(x)
        if len(cls_tokens) > 1:
            cls_tokens = torch.cat(cls_tokens, dim=1)
        else:
            cls_tokens = cls_tokens[0]
            
        # Optional: Late stage Fusion
        if self.combine_cls:
            cross_cls_token = self.cross_cls_token.expand(cls_tokens.shape[0], -1, -1)
            cls_tokens = torch.cat((cross_cls_token, cls_tokens), dim=1)

            # Apply MHSA across the CLS tokens, and only use the Cross-CLS token
            cls_tokens = self.cls_aggregator(cls_tokens, return_weights)
            cls_tokens = cls_tokens[:,0]
            
        else:
            cls_tokens = cls_tokens[:, -1]

        # Get logits
        return self.cls_fc(cls_tokens)

    def reset_tokenizer_grad(self):
        if self.shared_tokenizer:
            self.tokenizer.v.grad = None
        else:
            for key in self.backbone_feature_maps:
                self.tokenizer[key].v.grad = None

class ViT(nn.Module):
    """ 
    Based on Ross Wightman's ViT implementation: https://github.com/rwightman/pytorch-image-models/blob/dc422820eca4e550a4057561e595fc8b36209137/timm/models/vision_transformer.py
    """

    def __init__(self, token_dim, representation_size, block_type, block_drop, depth, norm_layer, act_layer, transformer_block_kwargs):
        super().__init__()
        
        self.token_dim = token_dim 
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        if depth > 1:
            dpr = [x.item() for x in torch.linspace(0, block_drop, depth)]  # stochastic depth decay rule
        else:
            dpr = [block_drop]
        self.blocks = nn.ModuleList([block_type(dim=token_dim, norm_layer=norm_layer, act_layer=act_layer, block_drop=dpr[i], **transformer_block_kwargs) for i in range(depth)])
        self.depth = depth
        
        self.norm = norm_layer(token_dim)

        # Representation layer
        if representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(token_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def forward(self, x, use_mean_token = False):

        # Go through transformer encoder blocks
        for i in range(self.depth):
            x = self.blocks[i](x)

        # Apply normalization layer
        x = self.norm(x)

        # Return representation of CLS token
        if use_mean_token:
            return self.pre_logits(x.mean(dim=1)), x
        else:
            return self.pre_logits(x[:, 0]), x


def _init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def _tresnet_init_weights(module: nn.Module):
    if isinstance(module, nn.Linear): 
        module.weight.data.normal_(0, 0.01)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def parameter_count(model, name):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainparams = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("{} - {}/{}".format(name, pytorch_total_trainparams, pytorch_total_params))

    return pytorch_total_trainparams, pytorch_total_params