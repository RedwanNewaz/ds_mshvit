import os
from argparse import ArgumentParser
from argparse import Namespace
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from metrics import F2CIW, F1Normal

import wandb

wandb.login()

from core import backbones
from core import heads
from core import tokenizers
from core import blocks
from core import optimizer
from core import scheduler
from core import class_weight
from core import datamodules
from core import transforms
from core import ema
from core import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def load_model(model_path, args):
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

    # load pretrained weights
    backbone.load_state_dict(updated_backbone_state_dict)
    head.load_state_dict(updated_head_state_dict)
    return backbone, head, bargs, hargs


def getTrainableModel(pargs):
    args = vars(pargs)
    opti_backbone, opti_head, _, _ = load_model(args["optical_model_path"], args)
    rgb_backbone, rgb_head, bargs, hargs = load_model(args["baseline_model_path"], args)
    return {
        'rgb_backbone': rgb_backbone,
        'rgb_head': rgb_head,
        'opti_backbone': opti_backbone,
        'opti_head': opti_head,
        'bargs' : bargs

    }


class Model(pl.LightningModule):
    def __init__(self, model, num_classes, loss_weight, sigmoid_loss, args):
        super().__init__()

        self.save_hyperparameters()

        self.rgb_backbone = model["rgb_backbone"].to(device)
        self.rgb_head = model["rgb_head"].to(device)
        self.opti_backbone = model["opti_backbone"].to(device)
        self.opti_head = model["opti_head"].to(device)

        self.criterion_labels = torch.nn.BCEWithLogitsLoss(weight=loss_weight)
        self.criterion_feats = torch.nn.BCEWithLogitsLoss()

        # Setup validation loss
        self.validation_criterion_labels = torch.nn.BCEWithLogitsLoss(weight=loss_weight)
        self.validation_criterion_feats = torch.nn.BCEWithLogitsLoss()

        # Setup accuracy logger
        self.rgb_valid_f2ciw = F2CIW()
        self.rgb_valid_f1normal = F1Normal()
        self.opti_valid_f2ciw = F2CIW()
        self.opti_valid_f1normal = F1Normal()

        self.all_time_f2ciw = 0.
        self.all_time_f1normal = 0.

        self.autocast_loss = True

    def forward(self, rgbX, optiX):
        rgb_feat_map = self.rgb_backbone(rgbX)
        opti_feat_map = self.opti_backbone(optiX)
        rgb_logits = self.rgb_head(rgb_feat_map)
        opti_logits = self.opti_head(opti_feat_map)
        return rgb_feat_map, opti_feat_map, rgb_logits, opti_logits

    def training_step(self, batch, batch_idx):
        rgb_imgs, opti_imgs, targets, _ = batch
        rgb_feat_maps, opti_feat_maps, rgb_logits, opti_logits = self(rgb_imgs, opti_imgs)

        with torch.cuda.amp.autocast(enabled=self.autocast_loss):
            label_loss = self.criterion_labels(rgb_logits, targets) + self.criterion_labels(opti_logits, targets)
            # layer3_loss = self.criterion_feats(rgb_feat_maps['layer3'], opti_feat_maps['layer3'])
            layer4_loss = torch.abs(self.criterion_feats(rgb_feat_maps['layer4'], opti_feat_maps['layer4']))
            # total_loss = label_loss + layer3_loss + layer4_loss
            total_loss = label_loss  + layer4_loss
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):

        rgb_imgs, opti_imgs, targets, _ = batch
        rgb_feat_maps, opti_feat_maps, rgb_logits, opti_logits = self(rgb_imgs, opti_imgs)

        with torch.cuda.amp.autocast(enabled=self.autocast_loss):
            label_loss = self.criterion_labels(rgb_logits, targets) + self.criterion_labels(opti_logits, targets)
            # layer3_loss = self.criterion_feats(rgb_feat_maps['layer3'], opti_feat_maps['layer3'])
            layer4_loss = torch.abs(self.criterion_feats(rgb_feat_maps['layer4'], opti_feat_maps['layer4']))
            # total_loss = label_loss + layer3_loss + layer4_loss
            total_loss = label_loss +  layer4_loss

        self.log('val_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.rgb_valid_f2ciw(rgb_logits.sigmoid(), targets)
        self.rgb_valid_f1normal(rgb_logits.sigmoid(), targets)

        self.opti_valid_f2ciw(opti_logits.sigmoid(), targets)
        self.opti_valid_f1normal(opti_logits.sigmoid(), targets)

        self.log('rgb_val_F2CIW', self.rgb_valid_f2ciw, on_step=False, on_epoch=True, prog_bar=False)
        self.log('rgb_val_F1Normal', self.rgb_valid_f1normal, on_step=False, on_epoch=True, prog_bar=False)
        self.log('opti_val_F2CIW', self.opti_valid_f2ciw, on_step=False, on_epoch=True, prog_bar=False)
        self.log('opti_val_F1Normal', self.opti_valid_f1normal, on_step=False, on_epoch=True, prog_bar=False)

        return total_loss

    def on_before_zero_grad(self, *args, **kwargs):
        # if self.hparams.args.model_ema:
        #     self.backbone_ema.update(self.backbone)
        #     self.head_ema.update(self.head)
        pass
    def on_after_backward(self, *args, **kwargs):
        if self.trainer.global_step < self.hparams.args.freeze_cluster_niters and self.sinkhorn_head:
            self.rgb_head.reset_tokenizer_grad()
            self.opti_head.reset_tokenizer_grad()

    def validation_epoch_end(self, *args, **kwargs):
        val_F2CIW = (self.trainer.logged_metrics["rgb_val_F2CIW"] + self.trainer.logged_metrics["opti_val_F2CIW"]) / 2.0
        val_F1Normal = (self.trainer.logged_metrics["rgb_val_F1Normal"] + self.trainer.logged_metrics["opti_val_F1Normal"]) / 2.0
        if self.hparams.args.model_ema and not self.hparams.args.model_ema_force_cpu:

            max_f2ciw = max(val_F2CIW, self.trainer.logged_metrics["val_F2CIW_ema"])
            max_f1normal = max(val_F1Normal,
                               self.trainer.logged_metrics["val_F1Normal_ema"])
        else:
            max_f2ciw = val_F2CIW
            max_f1normal = val_F1Normal
        self.log("valid_max_F2CIW", max_f2ciw, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("valid_max_F1Normal", max_f1normal, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        if max_f2ciw > self.all_time_f2ciw:
            self.all_time_f2ciw = max_f2ciw
        self.log("all_time_f2ciw", self.all_time_f2ciw, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        if max_f1normal > self.all_time_f1normal:
            self.all_time_f1normal = max_f1normal
        self.log("all_time_f1normal", self.all_time_f1normal, on_step=False, on_epoch=True, prog_bar=False,
                 sync_dist=True)

    def configure_optimizers(self):
        print("len(train_loader) {}".format(len(self.train_dataloader())))
        print("len(val_loader) {}".format(len(self.val_dataloader())))

        if self.hparams.args.freeze_layer_index != "None":
            for idx, child in enumerate(self.rgb_backbone.backbone.children()):
                if idx <= self.rgb_backbone.freeze_index[self.hparams.args.freeze_layer_index]:
                    for param in child.parameters():
                        param.requires_grad = False
            # repeat for opti
            for idx, child in enumerate(self.opti_backbone.backbone.children()):
                if idx <= self.opti_backbone.freeze_index[self.hparams.args.freeze_layer_index]:
                    for param in child.parameters():
                        param.requires_grad = False

        rgb_params_backbone = optimizer.adjusted_parameter_setting(self.rgb_backbone, self.hparams.args.lr,
                                                               self.hparams.args.weight_decay)
        rgb_params_head = optimizer.adjusted_parameter_setting(self.rgb_head, self.hparams.args.lr,
                                                           self.hparams.args.weight_decay)

        opti_params_backbone = optimizer.adjusted_parameter_setting(self.opti_backbone, self.hparams.args.lr,
                                                                   self.hparams.args.weight_decay)
        opti_params_head = optimizer.adjusted_parameter_setting(self.opti_head, self.hparams.args.lr,
                                                               self.hparams.args.weight_decay)
        params = rgb_params_backbone + rgb_params_head + opti_params_backbone + opti_params_head

        opt_args = {"optim": "SGD", "lr": 0., "weight_decay": 0., "momentum": self.hparams.args.momentum,
                    "nesterov": self.hparams.args.nesterov}

        scheduler_args = {"lr_schedule": "Step",
                          "schedule_int": "epoch",
                          "lr_steps": self.hparams.args.lr_steps,
                          "lr_gamma": self.hparams.args.lr_gamma
                          }

        optim = optimizer.get_optimizer(params, opt_args)
        sched = scheduler.get_lr_scheduler(optim, scheduler_args)

        sched = {"scheduler": sched,
                 "interval": "epoch",
                 "frequency": 1}

        return [optim], [sched]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Optimization")
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--nesterov', action='store_true')
        parser.add_argument('--lr_gamma', type=float, default=0.01)
        parser.add_argument('--lr_steps', nargs='+', type=int, default=[20, 30])
        parser.add_argument('--effective_beta', type=float, default=0.9999)

        parser = parent_parser.add_argument_group("Backbone")
        parser.add_argument('--backbone_model', type=str, choices=backbones.VALID_BACKBONES)
        parser.add_argument('--img_size', type=int, choices=[224, 299, 336, 384, 448, 576, 640])
        parser.add_argument('--pretrained_backbone', type=str, default="")
        parser.add_argument('--backbone_feature_maps', nargs='+', type=str,
                            choices=["stem", "layer1", "layer2", "layer3", "layer4"])
        parser.add_argument('--freeze_layer_index', type=str, default="None",
                            choices=["None", "stem", "layer1", "layer2", "layer3", "layer4"])
        parser.add_argument('--head_model', type=str, choices=heads.HEADS)

        parser = parent_parser.add_argument_group("BaseHead")
        parser.add_argument('--sigmoid_loss', action='store_true')
        parser.add_argument('--tresnet_init', action='store_true')
        parser.add_argument('--global_pool', type=str, default="avg", choices=["avg", "max"])

        parser = parent_parser.add_argument_group("MultiScaleViT")
        parser.add_argument('--use_mean_token', action='store_true')
        parser.add_argument('--freeze_cluster_niters', type=int, default=0)
        parser.add_argument('--token_dim', type=int)
        parser.add_argument('--norm_layer', default=None)
        parser.add_argument('--act_layer', default=None)
        parser.add_argument('--transformer_depth', nargs='+', type=int)
        parser.add_argument('--representation_size', type=int, default=None)
        parser.add_argument('--block_drop', type=float)
        parser.add_argument('--tokenizer_layer', type=str, default="Patchify", choices=tokenizers.TOKENIZERS)
        parser.add_argument('--block_type', type=str, default="TransformerBlock", choices=blocks.BLOCKS)
        parser.add_argument('--use_pos_embed', action='store_true')
        parser.add_argument('--num_heads', nargs='+', type=int)
        parser.add_argument('--qkv_bias', action='store_true')
        parser.add_argument('--mlp_ratio', nargs='+', type=float)
        parser.add_argument('--proj_drop', nargs='+', type=float)
        parser.add_argument('--attn_drop', nargs='+', type=float)
        parser.add_argument('--pos_embed_drop', type=float)
        parser.add_argument('--shared_tower', action='store_true')

        parser.add_argument('--multiscale_method', type=str,
                            choices=["SharedTokenizer", "SeparateScale", "CrossScale-CNN", "CrossScale-Token",
                                     "CrossScale-VIT"])
        parser.add_argument('--late_fusion', action='store_true')
        parser.add_argument('--cross_scale_all', action='store_true')
        parser.add_argument('--shared_tokenizer', action='store_true')
        parser.add_argument('--no_vit_layers', nargs='+', type=str,
                            choices=["stem", "layer1", "layer2", "layer3", "layer4"])

        parser.add_argument('--cross_block_type', type=str, default="MHSABlock", choices=blocks.BLOCKS)
        parser.add_argument('--cross_num_heads', type=int)
        parser.add_argument('--cross_mlp_ratio', type=float)
        parser.add_argument('--cross_qkv_bias', action='store_true')
        parser.add_argument('--cross_proj_drop', type=float)
        parser.add_argument('--cross_attn_drop', type=float)
        parser.add_argument('--cross_block_drop', type=float)

        parser = parent_parser.add_argument_group("Tokenizers")
        parser.add_argument('--num_clusters', nargs='+', type=int)
        parser.add_argument('--l2_normalize', action='store_true')
        parser.add_argument('--patch_size', nargs='+', type=int)
        parser.add_argument('--sinkhorn_eps', nargs='+', type=float)
        parser.add_argument('--sinkhorn_iters', nargs='+', type=int)

        # Model Exponential Moving Average
        parser = parent_parser.add_argument_group("Model EMA")
        parser.add_argument('--model_ema', action='store_true')
        parser.add_argument('--model_ema_force_cpu', action='store_true')
        parser.add_argument('--model_ema_decay', type=float, default=0.9997)

        return parent_parser

def main(args):
    args.seed = pl.seed_everything(args.seed)
    preTrainedModels = getTrainableModel(args)
    bargs = preTrainedModels["bargs"]

    train_transform = transforms.create_sewerml_train_transformations(
        {"img_size": bargs.img_size, "model_name": bargs.backbone_model})
    eval_transform = transforms.create_sewerml_eval_transformations(
        {"img_size": bargs.img_size, "model_name": bargs.backbone_model})


    dm = datamodules.get_joint_datamodule(dataset=args.dataset, batch_size=args.batch_size, workers=args.workers,
                                    ann_root=args.ann_root, rgb_data_root=args.rgb_data_root, opti_data_root=args.opti_data_root, train_transform=train_transform,
                                    eval_transform=eval_transform)

    dm.prepare_data()
    dm.setup("fit")

    weights = class_weight.effective_samples(dm.train_dataset.labels, dm.num_classes, args.effective_beta)


    model = Model(preTrainedModels, num_classes=dm.num_classes, loss_weight=weights, sigmoid_loss=args.sigmoid_loss, args=args)

    # Setup Logger
    version = "version_" + str(args.log_version)
    model_name = args.dataset + "_" + args.backbone_model + "_" + args.head_model + "_" + args.block_type + "_" + args.tokenizer_layer
    print("-" * 15 + model_name + "-" * 15)

    os.makedirs(args.log_save_dir, exist_ok=True)
    logger_path = os.path.join(args.log_save_dir, model_name, "version_" + str(args.log_version))
    os.makedirs(logger_path, exist_ok=True)

    logger = WandbLogger(project=args.wandb_project,  # group runs in "MNIST" project
                         log_model='all',
                         save_dir=logger_path,
                         version=version,
                         name=model_name,
                         **{"group": args.wandb_group})  # log all new checkpoints during training



    if args.monitor_metric:
        monitor = "valid_max_F1Normal"
        filename = '{epoch:02d}-{valid_max_F1Normal:.4f}'
        mode = "max"

    else:
        monitor = "val_loss"
        filename = '{epoch:02d}-{val_loss:.4f}'
        mode = "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger_path),
        filename=filename,
        save_top_k=args.save_top_k,
        save_last=args.save_last,
        verbose=False,
        monitor=monitor,
        mode=mode,
        every_n_val_epochs=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [checkpoint_callback, lr_monitor]

    if args.deterministic:
        args.benchmark = False
    trainer = pl.Trainer.from_argparse_args(args, terminate_on_nan=True, logger=logger, callbacks=callbacks)

    try:
        trainer.fit(model, dm)
    except Exception as e:
        print(e)
        with open(os.path.join(logger_path, "error.txt"), "w") as f:
            f.write(str(e))

def run_cli():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument('--conda_env', type=str, default='')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default='./annotations_sewerml')

    # my modifications
    parser.add_argument('--rgb_data_root', type=str, default='')
    parser.add_argument('--opti_data_root', type=str, default='')
    parser.add_argument("--baseline_model_path", type=str)
    parser.add_argument("--optical_model_path", type=str)

    parser.add_argument('--batch_size', type=int, default=64, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--log_save_dir', type=str, default="")
    parser.add_argument('--log_version', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataset', type=str, choices=["SewerMLJoint"], default="SewerMLJoint")
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--wandb_group', type=str, default="")
    parser.add_argument('--monitor_metric', action='store_true')
    parser.add_argument('--save_last', action='store_true')
    parser.add_argument('--save_top_k', type=int, default=1)

    parser.add_argument("--use_ema", action='store_true')
    parser.add_argument("--filter_novit", action='store_true')
    parser.add_argument("--model_version", type=str, default="")

    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    run_cli()