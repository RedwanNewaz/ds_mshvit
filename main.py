# !/home/aredwann/anaconda3/envs/MSHViT3/bin/python
from argparse import ArgumentParser
from argparse import Namespace

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from core import backbones
from core import heads
from core import blocks
from core import dataloader
from core import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, args):
    outputPath = args["results_output"]
    use_ema = args["use_ema"]
    filter_novit = args["filter_novit"]

    if not os.path.isfile(model_path):
        raise ValueError("The provided path is not a file: {}".format(model_path))
    try:
        print(model_path)
        best_model = torch.load(model_path, map_location=device)
    except Exception as e:
        print(str(e))
        return None

    bargs = Namespace(**best_model["hyper_parameters_backbone"])
    hargs = Namespace(**best_model["hyper_parameters_head"])

    if bargs.backbone_model in backbones.VALID_BACKBONES:
        backbone = backbones.get_backbone(backbone_model=bargs.backbone_model,
                                          img_size=bargs.img_size,
                                          return_stages=bargs.backbone_feature_maps,
                                          pretrained_backbone=bargs.pretrained_backbone,
                                          custom_pretrained_path="")

    if hargs.head_model in heads.HEADS:
        if hargs.head_model == "BaseHead":
            head = heads.BaseHead(num_classes=hargs.num_classes,
                                  backbone_feature_map_sizes=backbone.feature_map_sizes,
                                  last_stage=backbone.last_stage,
                                  global_pool=hargs.global_pool,
                                  sigmoid_loss=hargs.sigmoid_loss,
                                  tresnet_init=hargs.tresnet_init)

        elif hargs.head_model == "MultiScaleViTHead":
            tokenizer_kwargs = {"patch_size": hargs.patch_size,
                                "num_clusters": hargs.num_clusters,
                                "l2_normalize": hargs.l2_normalize,
                                "eps": hargs.sinkhorn_eps,
                                "iters": hargs.sinkhorn_iters}

            transformer_block_kwargs = {"num_heads": hargs.num_heads,
                                        "qkv_bias": hargs.qkv_bias,
                                        "mlp_ratio": hargs.mlp_ratio,
                                        "proj_drop": hargs.proj_drop,
                                        "attn_drop": hargs.attn_drop}

            cross_attention_kwargs = {"num_heads": hargs.cross_num_heads,
                                      "qkv_bias": hargs.cross_qkv_bias,
                                      "mlp_ratio": hargs.cross_mlp_ratio,
                                      "proj_drop": hargs.cross_proj_drop,
                                      "attn_drop": hargs.cross_attn_drop,
                                      "block_drop": hargs.cross_block_drop}

            if filter_novit and hargs.multiscale_method in ["CrossScale-Token", "CrossScale-CNN"]:
                no_vit_layers = [m for m in hargs.backbone_feature_maps if m != "layer4"]
            else:
                no_vit_layers = hargs.no_vit_layers

            head = heads.MultiScaleViTHead(num_classes=hargs.num_classes,
                                           token_dim=hargs.token_dim,
                                           representation_size=hargs.representation_size,
                                           tokenizer_layer_name=hargs.tokenizer_layer,
                                           block_type=blocks.__dict__[hargs.block_type],
                                           block_drop=hargs.block_drop,
                                           use_pos_embed=hargs.use_pos_embed,
                                           pos_embed_drop=hargs.pos_embed_drop,
                                           backbone_feature_map_sizes=backbone.feature_map_sizes,
                                           backbone_feature_maps=hargs.backbone_feature_maps,
                                           transformer_depths=hargs.transformer_depth,
                                           sigmoid_loss=hargs.sigmoid_loss,
                                           norm_layer=hargs.norm_layer,
                                           act_layer=hargs.act_layer,
                                           tokenizer_kwargs_base=tokenizer_kwargs,
                                           transformer_block_kwargs_base=transformer_block_kwargs,
                                           cross_attention_kwargs=cross_attention_kwargs,
                                           cross_block_type=blocks.__dict__[hargs.cross_block_type],
                                           shared_tower=hargs.shared_tower,
                                           multiscale_method=hargs.multiscale_method,
                                           late_fusion=hargs.late_fusion,
                                           cross_scale_all=hargs.cross_scale_all,
                                           no_vit_layers=no_vit_layers,
                                           shared_tokenizer=hargs.shared_tokenizer,
                                           tresnet_init=hargs.tresnet_init,
                                           use_mean_token=hargs.use_mean_token)

        else:
            raise ValueError("Got head {}, but no such head is in this codebase".format(hargs.head_model))
    else:
        raise ValueError("Got head {}, but no such head is in this codebase".format(hargs.head_model))

    model_version = args["model_version"]
    if model_version == "":
        model_version = os.path.splitext(os.path.basename(model_path))[0]
    if use_ema:
        model_version += "_EMA"
    if filter_novit:
        model_version += "_NOViT"

    # Load best checkpoint
    if use_ema:
        print("EMA")
        updated_backbone_state_dict = best_model["state_dict_backbone_ema"]
        updated_head_state_dict = best_model["state_dict_head_ema"]
    else:
        print("NO EMA")
        updated_backbone_state_dict = best_model["state_dict_backbone"]
        updated_head_state_dict = best_model["state_dict_head"]

    backbone.load_state_dict(updated_backbone_state_dict)
    head.load_state_dict(updated_head_state_dict)
    return backbone, head, bargs, hargs

def getTrainableModel(args):
    opti_backbone, opti_head, _, _ = load_model(args["optical_model_path"], args)
    rgb_backbone, rgb_head, bargs, hargs = load_model( args["baseline_model_path"], args)
    train_transform = transforms.create_sewerml_eval_transformations(
        {"img_size": bargs.img_size, "model_name": bargs.backbone_model})
    train_dataloader, label_names = dataloader.get_joint_dataloader(args["dataset"], args["batch_size"],
                                                                    args["workers"],args['ann_root'],
                                                                    args["rgb_data_root"], args["opti_data_root"],
                                                                    "Train", train_transform)
    return {
        'rgb_backbone': rgb_backbone,
        'rgb_head': rgb_head,
        'opti_backbone': opti_backbone,
        'opti_head': opti_head,
        'dataloader': train_dataloader,
        'labels': label_names
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='Pytorch-Lightning')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default='./annotations_sewerml')
    parser.add_argument('--dataset', type=str, choices=["SewerML"], default="SewerML")
    parser.add_argument('--rgb_data_root', type=str, default='')
    parser.add_argument('--opti_data_root', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=512, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument("--baseline_model_path", type=str)
    parser.add_argument("--optical_model_path", type=str)
    parser.add_argument("--model_version", type=str, default="")
    parser.add_argument("--results_output", type=str, default="")
    parser.add_argument("--eval_val", action='store_true')
    parser.add_argument("--eval_test", action='store_true')
    parser.add_argument("--use_ema", action='store_true')
    parser.add_argument("--filter_novit", action='store_true')

    args = vars(parser.parse_args())

    model = getTrainableModel(args)

    rbg_backbone = model["rgb_backbone"].to(device)
    rgb_head = model["rgb_head"].to(device)
    opti_backbone = model["opti_backbone"].to(device)
    opti_head = model["opti_head"].to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    act_func = nn.Sigmoid()

    for i, (rbg_imgs, opti_imgs, targets, path) in enumerate(model["dataloader"]):

        rbg_imgs = rbg_imgs.to(device)
        opti_imgs = opti_imgs.to(device)
        targets = targets.to(device)

        rgb_feat_maps = rbg_backbone(rbg_imgs)
        rgb_outputs = act_func(rgb_head(rgb_feat_maps, True, True))
        opti_feat_maps = rbg_backbone(opti_imgs)
        opti_outputs = act_func(rgb_head(opti_feat_maps, True, True))

        rgbLoss = criterion(rgb_outputs, targets)
        optLoss = criterion(opti_outputs, targets)
        layer3_loss = criterion(rgb_feat_maps['layer3'], opti_feat_maps['layer3'])
        layer4_loss = criterion(rgb_feat_maps['layer4'], opti_feat_maps['layer4'])

        total_loss = rgbLoss + optLoss + layer3_loss + layer4_loss
        print(total_loss)

        if i > 4:
            break

