
TV_RESNET_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2"]
TRESNET_MODELS = ["tresnet_m", "tresnet_l", "tresnet_xl"]
BOTNET_MODELS = ["botnet"]
COATNET_MODELS = ["coatnet0", "coatnet1"]

TV_MODELS = TV_RESNET_MODELS

import torch
import torch.nn as nn

from torchvision.models.resnet import ResNet
import torchvision.models as torch_models

VALID_BACKBONES = [] + TV_MODELS
try:
    from nets.tresnet import TResnetM, TResnetL, TResnetXL, TResNet
    VALID_BACKBONES +=  TRESNET_MODELS  
except ImportError as e:
    print("NO TRESNET")
    
try:
    from nets.botnet import BottleStack
    VALID_BACKBONES +=  BOTNET_MODELS  
except ImportError as e:
    print("NO BOTNET")
    
try:
    from nets.coatnet import CoAtNet
    VALID_BACKBONES +=  COATNET_MODELS  
except ImportError as e:
    print("NO COATNET")





class FrozenBatchNorm2d(torch.nn.Module):
    # From https://github.com/SlongLiu/query2labels/blob/5be909747984b4c6c62de837ebb035caa7e31b33/lib/models/backbone.py#L80
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class TV_ResNet(nn.Module):
    def __init__(self, backbone_model, img_size, return_stages, pretrained_backbone=False, use_frozen_backbone = False, **kwargs):
        super().__init__()

        if pretrained_backbone and use_frozen_backbone:
            norm_layer = FrozenBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.backbone = torch_models.__dict__[backbone_model](pretrained=pretrained_backbone, norm_layer = norm_layer, **kwargs)

        assert(isinstance(self.backbone, ResNet))
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.last_stage = "layer4"

        self.freeze_index = {"stem": 3, "layer1": 4, "layer2": 5, "layer3": 6, "layer4": 9} # The index at which the last layer will be frozen

        if return_stages:
            self.return_stages = return_stages
        else:
            self.return_stages = [self.last_stage]

        self.feature_map_sizes = {}
        with torch.no_grad():
            training = self.backbone.training
            if training:
                self.backbone.eval()
            dummy_input = torch.zeros(1, 3, img_size, img_size)

            o0 = self.forward_stage(dummy_input, "stem")
            o1 = self.forward_stage(o0, "layer1")
            o2 = self.forward_stage(o1, "layer2")
            o3 = self.forward_stage(o2, "layer3")
            o4 = self.forward_stage(o3, "layer4")

            self.feature_map_sizes["stem"] = [o0.shape[-1], o0.shape[1]]
            self.feature_map_sizes["layer1"] = [o1.shape[-1], o1.shape[1]]
            self.feature_map_sizes["layer2"] = [o2.shape[-1], o2.shape[1]]
            self.feature_map_sizes["layer3"] = [o3.shape[-1], o3.shape[1]]
            self.feature_map_sizes["layer4"] = [o4.shape[-1], o4.shape[1]]

            self.backbone.train(training)    

    def forward_stage(self, x, stage):
        assert(stage in ["stem", 'layer1','layer2','layer3','layer4'])
        if stage == 'stem':
            x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
            x = self.backbone.maxpool(x)
            return x

        else: # Stage 1, 2, 3 or 4
            layer = getattr(self.backbone, stage)
            return layer(x)

    def forward(self, x):
        feat_maps = {}
        x0 = self.forward_stage(x, "stem")
        x1 = self.forward_stage(x0, "layer1")
        x2 = self.forward_stage(x1, "layer2")
        x3 = self.forward_stage(x2, "layer3")
        x4 = self.forward_stage(x3, "layer4")

        if "stem" in self.return_stages:
            feat_maps["stem"] = x0
        if "layer1" in self.return_stages:
            feat_maps["layer1"] = x1
        if "layer2" in self.return_stages:
            feat_maps["layer2"] = x2
        if "layer3" in self.return_stages:
            feat_maps["layer3"] = x3
        if "layer4" in self.return_stages:
            feat_maps["layer4"] = x4

        return feat_maps

class MIIL_TResNet(nn.Module):
    def __init__(self, backbone_model, img_size, return_stages, pretrained_backbone="", **kwargs):
        super().__init__()
        
        model_params = {"num_classes": 1}
        if backbone_model.lower() == 'tresnet_m':
            self.backbone = TResnetM(model_params)
        elif backbone_model.lower() == 'tresnet_l':
            self.backbone = TResnetL(model_params)
        elif backbone_model.lower() == 'tresnet_xl':
            self.backbone = TResnetXL(model_params)

        if pretrained_backbone != "":
            state = torch.load(pretrained_backbone, map_location='cpu')
            if "state_dict" in list(state.keys()):
                key_name = "state_dict"
            else:
                key_name = "model"
            filtered_dict = {k: v for k, v in state[key_name].items() if
                            (k in self.backbone.state_dict() and 'head.fc' not in k)}
            self.backbone.load_state_dict(filtered_dict, strict=False)
            print("Loaded: {}".format(pretrained_backbone))
            
        assert(isinstance(self.backbone, TResNet))
        self.backbone.global_pool = nn.Identity()
        self.backbone.head = nn.Identity()
        self.backbone = self.backbone.body

        self.last_stage = "layer4"

        self.freeze_index = {"stem": 1, "layer1": 2, "layer2": 3, "layer3": 4, "layer4": 5} # The index at which the last layer will be frozen

        if return_stages:
            self.return_stages = return_stages
        else:
            self.return_stages = [self.last_stage]

        self.feature_map_sizes = {}
        with torch.no_grad():
            training = self.backbone.training
            if training:
                self.backbone.eval()
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            
            o0 = self.forward_stage(dummy_input, "stem")
            o1 = self.forward_stage(o0, "layer1")
            o2 = self.forward_stage(o1, "layer2")
            o3 = self.forward_stage(o2, "layer3")
            o4 = self.forward_stage(o3, "layer4")

            self.feature_map_sizes["stem"] = [o0.shape[-1], o0.shape[1]]
            self.feature_map_sizes["layer1"] = [o1.shape[-1], o1.shape[1]]
            self.feature_map_sizes["layer2"] = [o2.shape[-1], o2.shape[1]]
            self.feature_map_sizes["layer3"] = [o3.shape[-1], o3.shape[1]]
            self.feature_map_sizes["layer4"] = [o4.shape[-1], o4.shape[1]]

            self.backbone.train(training)    

    def forward_stage(self, x, stage):
        assert(stage in ["stem", 'layer1','layer2','layer3','layer4'])
        if stage == 'stem':
            x = self.backbone[:2](x) #SpaceToDepth, conv1
        elif stage == "layer1":
            x = self.backbone[2](x) #layer1
        elif stage == "layer2":
            x = self.backbone[3](x) #layer2
        elif stage == "layer3":
            x = self.backbone[4](x) #layer3
        elif stage == "layer4":
            x = self.backbone[5](x) #layer4
        return x

    def forward(self, x):
        feat_maps = {}
        x0 = self.forward_stage(x, "stem")
        x1 = self.forward_stage(x0, "layer1")
        x2 = self.forward_stage(x1, "layer2")
        x3 = self.forward_stage(x2, "layer3")
        x4 = self.forward_stage(x3, "layer4")

        if "stem" in self.return_stages:
            feat_maps["stem"] = x0
        if "layer1" in self.return_stages:
            feat_maps["layer1"] = x1
        if "layer2" in self.return_stages:
            feat_maps["layer2"] = x2
        if "layer3" in self.return_stages:
            feat_maps["layer3"] = x3
        if "layer4" in self.return_stages:
            feat_maps["layer4"] = x4

        return feat_maps

class GitHub_BoTNet(nn.Module):
    def __init__(self, backbone_model, img_size, return_stages, pretrained_backbone=False, use_frozen_backbone = False, **kwargs):
        super().__init__()
        
        if pretrained_backbone and use_frozen_backbone:
            norm_layer = FrozenBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        resnet = torch_models.__dict__["resnet50"](pretrained=pretrained_backbone, norm_layer = norm_layer, **kwargs)

        # BoTNet Layer
        botnet_layer = BottleStack(
            dim = 1024,
            fmap_size = 14,       # Output size of layer3
            dim_out = 2048,
            proj_factor = 4,
            downsample = True,
            heads = 4,
            dim_head = 128,
            rel_pos_emb = True,
            activation = nn.ReLU(),
            num_layers = 3
        )
        
        resnet_layers = list(resnet.children())
        self.backbone = nn.Sequential(
            *resnet_layers[:-3],
            botnet_layer
        )

        del resnet
        del resnet_layers
        
        self.last_stage = "layer4"

        self.freeze_index = {"stem": 3, "layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7} # The index at which the last layer will be frozen

        if return_stages:
            self.return_stages = return_stages
        else:
            self.return_stages = [self.last_stage]

        self.feature_map_sizes = {}
        with torch.no_grad():
            training = self.backbone.training
            if training:
                self.backbone.eval()
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            
            o0 = self.forward_stage(dummy_input, "stem")
            o1 = self.forward_stage(o0, "layer1")
            o2 = self.forward_stage(o1, "layer2")
            o3 = self.forward_stage(o2, "layer3")
            o4 = self.forward_stage(o3, "layer4")

            self.feature_map_sizes["stem"] = [o0.shape[-1], o0.shape[1]]
            self.feature_map_sizes["layer1"] = [o1.shape[-1], o1.shape[1]]
            self.feature_map_sizes["layer2"] = [o2.shape[-1], o2.shape[1]]
            self.feature_map_sizes["layer3"] = [o3.shape[-1], o3.shape[1]]
            self.feature_map_sizes["layer4"] = [o4.shape[-1], o4.shape[1]]

            self.backbone.train(training)    

    def forward_stage(self, x, stage):
        assert(stage in ["stem", 'layer1','layer2','layer3','layer4'])
        if stage == 'stem':
            x = self.backbone[:4](x) #conv2, bn, relu, maxpool
        elif stage == "layer1":
            x = self.backbone[4](x) #layer1
        elif stage == "layer2":
            x = self.backbone[5](x) #layer2
        elif stage == "layer3":
            x = self.backbone[6](x) #layer3
        elif stage == "layer4":
            x = self.backbone[7:](x) #layer4
        return x

    def forward(self, x):
        feat_maps = {}
        x0 = self.forward_stage(x, "stem")
        x1 = self.forward_stage(x0, "layer1")
        x2 = self.forward_stage(x1, "layer2")
        x3 = self.forward_stage(x2, "layer3")
        x4 = self.forward_stage(x3, "layer4")

        if "stem" in self.return_stages:
            feat_maps["stem"] = x0
        if "layer1" in self.return_stages:
            feat_maps["layer1"] = x1
        if "layer2" in self.return_stages:
            feat_maps["layer2"] = x2
        if "layer3" in self.return_stages:
            feat_maps["layer3"] = x3
        if "layer4" in self.return_stages:
            feat_maps["layer4"] = x4

        return feat_maps

class GitHub_CoAtNet(nn.Module):
    def __init__(self, backbone_model, img_size, return_stages, pretrained_backbone=False, use_frozen_backbone = False, **kwargs):
        super().__init__()
        
        
        if backbone_model == "coatnet0":
            num_blocks = [2, 2, 3, 5, 2]            # L
            channels = [64, 96, 192, 384, 768]      # D
        elif backbone_model == "coatnet1":
            num_blocks = [2, 2, 6, 14, 2]            # L
            channels = [64, 96, 192, 384, 768]      # D

        self.backbone = CoAtNet((img_size, img_size), 3, num_blocks, channels)

        self.last_stage = "layer4"

        self.freeze_index = {"stem": 3, "layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7} # The index at which the last layer will be frozen

        if return_stages:
            self.return_stages = return_stages
        else:
            self.return_stages = [self.last_stage]

        self.feature_map_sizes = {}
        with torch.no_grad():
            training = self.backbone.training
            if training:
                self.backbone.eval()
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            
            o0 = self.forward_stage(dummy_input, "stem")
            o1 = self.forward_stage(o0, "layer1")
            o2 = self.forward_stage(o1, "layer2")
            o3 = self.forward_stage(o2, "layer3")
            o4 = self.forward_stage(o3, "layer4")

            self.feature_map_sizes["stem"] = [o0.shape[-1], o0.shape[1]]
            self.feature_map_sizes["layer1"] = [o1.shape[-1], o1.shape[1]]
            self.feature_map_sizes["layer2"] = [o2.shape[-1], o2.shape[1]]
            self.feature_map_sizes["layer3"] = [o3.shape[-1], o3.shape[1]]
            self.feature_map_sizes["layer4"] = [o4.shape[-1], o4.shape[1]]

            self.backbone.train(training)    

    def forward_stage(self, x, stage):
        assert(stage in ["stem", 'layer1','layer2','layer3','layer4'])
        if stage == 'stem':
            x = self.backbone.s0(x) #conv2, bn, relu, maxpool
        elif stage == "layer1":
            x = self.backbone.s1(x) #layer1
        elif stage == "layer2":
            x = self.backbone.s2(x) #layer2
        elif stage == "layer3":
            x = self.backbone.s3(x) #layer3
        elif stage == "layer4":
            x = self.backbone.s4(x) #layer4
        return x

    def forward(self, x):
        feat_maps = {}
        x0 = self.forward_stage(x, "stem")
        x1 = self.forward_stage(x0, "layer1")
        x2 = self.forward_stage(x1, "layer2")
        x3 = self.forward_stage(x2, "layer3")
        x4 = self.forward_stage(x3, "layer4")

        if "stem" in self.return_stages:
            feat_maps["stem"] = x0
        if "layer1" in self.return_stages:
            feat_maps["layer1"] = x1
        if "layer2" in self.return_stages:
            feat_maps["layer2"] = x2
        if "layer3" in self.return_stages:
            feat_maps["layer3"] = x3
        if "layer4" in self.return_stages:
            feat_maps["layer4"] = x4

        return feat_maps


def get_backbone(backbone_model, img_size, return_stages, pretrained_backbone=False, custom_pretrained_path="", **kwargs):
    
    if backbone_model not in TRESNET_MODELS:
        if pretrained_backbone != "":
            pretrained_backbone = True
        else:
            pretrained_backbone = False
    
    print("{} - Pretrained = {}".format(backbone_model, pretrained_backbone))

    if backbone_model in TV_RESNET_MODELS:
        backbone = TV_ResNet(backbone_model = backbone_model,
                         img_size = img_size,
                         return_stages = return_stages,
                         pretrained_backbone = pretrained_backbone)
    elif backbone_model in TRESNET_MODELS:
        backbone = MIIL_TResNet(backbone_model = backbone_model,
                                img_size = img_size,
                                return_stages = return_stages,
                                pretrained_backbone = "")
    elif backbone_model in BOTNET_MODELS:
        backbone = GitHub_BoTNet(backbone_model = backbone_model,
                                    img_size = img_size,
                                    return_stages = return_stages,
                                    pretrained_backbone = pretrained_backbone)
    elif backbone_model in COATNET_MODELS:
        backbone = GitHub_CoAtNet(backbone_model = backbone_model,
                                    img_size = img_size,
                                    return_stages = return_stages,
                                    pretrained_backbone = pretrained_backbone)
    else:
        raise ValueError("Got backbone {}, but no such backbone is in this codebase".format(backbone_model))

    if custom_pretrained_path != "":
        checkpoint = torch.load(custom_pretrained_path)
        backbone.load_state_dict(checkpoint['model_state_dict'])

    return backbone


def parameter_count(model, name):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainparams = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("{} - {}/{}".format(name, pytorch_total_trainparams, pytorch_total_params))

    return pytorch_total_trainparams, pytorch_total_params