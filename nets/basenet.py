from core import backbones
from core import heads
from core import blocks

class BaseNet:
    def __init__(self, num_classes, sigmoid_loss, pretrained_backbone, args):
        self.__setArchitecure(num_classes, sigmoid_loss, pretrained_backbone, args)

    def getArchi(self):
        return self.backbone, self.head

    def __setArchitecure(self, num_classes, sigmoid_loss, pretrained_backbone, args):
        # TODO pretrained_backbone remove to empty string
        self.backbone = backbones.get_backbone(backbone_model=args.backbone_model,
                                               img_size=args.img_size,
                                               return_stages=args.backbone_feature_maps,
                                               pretrained_backbone="",
                                               custom_pretrained_path="")

        if args.head_model in heads.HEADS:
            if args.head_model == "BaseHead":
                self.head = heads.BaseHead(num_classes=num_classes,
                                           backbone_feature_map_sizes=self.backbone.feature_map_sizes,
                                           last_stage=self.backbone.last_stage,
                                           global_pool=args.global_pool,
                                           sigmoid_loss=sigmoid_loss,
                                           tresnet_init=args.tresnet_init)
            elif args.head_model == "MultiScaleViTHead":
                tokenizer_kwargs = {"patch_size": args.patch_size,
                                    "num_clusters": args.num_clusters,
                                    "l2_normalize": args.l2_normalize,
                                    "eps": args.sinkhorn_eps,
                                    "iters": args.sinkhorn_iters}

                transformer_block_kwargs = {"num_heads": args.num_heads,
                                            "qkv_bias": args.qkv_bias,
                                            "mlp_ratio": args.mlp_ratio,
                                            "proj_drop": args.proj_drop,
                                            "attn_drop": args.attn_drop}

                cross_attention_kwargs = {"num_heads": args.cross_num_heads,
                                          "qkv_bias": args.cross_qkv_bias,
                                          "mlp_ratio": args.cross_mlp_ratio,
                                          "proj_drop": args.cross_proj_drop,
                                          "attn_drop": args.cross_attn_drop,
                                          "block_drop": args.cross_block_drop}

                self.head = heads.MultiScaleViTHead(num_classes=num_classes,
                                                    token_dim=args.token_dim,
                                                    representation_size=args.representation_size,
                                                    tokenizer_layer_name=args.tokenizer_layer,
                                                    block_type=blocks.__dict__[args.block_type],
                                                    block_drop=args.block_drop,
                                                    use_pos_embed=args.use_pos_embed,
                                                    pos_embed_drop=args.pos_embed_drop,
                                                    backbone_feature_map_sizes=self.backbone.feature_map_sizes,
                                                    backbone_feature_maps=args.backbone_feature_maps,
                                                    transformer_depths=args.transformer_depth,
                                                    sigmoid_loss=sigmoid_loss,
                                                    norm_layer=args.norm_layer,
                                                    act_layer=args.act_layer,
                                                    tokenizer_kwargs_base=tokenizer_kwargs,
                                                    transformer_block_kwargs_base=transformer_block_kwargs,
                                                    cross_attention_kwargs=cross_attention_kwargs,
                                                    cross_block_type=blocks.__dict__[args.cross_block_type],
                                                    shared_tower=args.shared_tower,
                                                    multiscale_method=args.multiscale_method,
                                                    late_fusion=args.late_fusion,
                                                    cross_scale_all=args.cross_scale_all,
                                                    no_vit_layers=args.no_vit_layers,
                                                    shared_tokenizer=args.shared_tokenizer,
                                                    tresnet_init=args.tresnet_init,
                                                    use_mean_token=args.use_mean_token)
            else:
                raise ValueError("Got head {}, but no such head is in this codebase".format(args.head_model))
        else:
            raise ValueError("Got head {}, but no such head is in this codebase".format(args.head_model))
