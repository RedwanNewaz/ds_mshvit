#!/home/aredwann/anaconda3/envs/MSHViT3/bin/python
import os
from argparse import ArgumentParser

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


class Model(pl.LightningModule):
    def __init__(self, num_classes, loss_weight, sigmoid_loss, args):
        super().__init__()

        self.save_hyperparameters()

        self.backbone = backbones.get_backbone(backbone_model=args.backbone_model,
                                               img_size=args.img_size,
                                               return_stages=args.backbone_feature_maps,
                                               pretrained_backbone=args.pretrained_backbone,
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

        # Main training loss setup
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=loss_weight)

        self.autocast_loss = True
        self.sinkhorn_head = args.head_model == "MultiScaleViTHead" and "Sinkhorn" in args.tokenizer_layer

        # Setup validation loss
        self.validation_criterion = torch.nn.BCEWithLogitsLoss(weight=loss_weight)

        # Setup accuracy logger
        self.valid_f2ciw = F2CIW()
        self.valid_f1normal = F1Normal()

        if args.model_ema:
            self.backbone_ema = ema.ModelEMA(self.backbone, decay=args.model_ema_decay,
                                             device='cpu' if args.model_ema_force_cpu else None)
            self.head_ema = ema.ModelEMA(self.head, decay=args.model_ema_decay,
                                         device='cpu' if args.model_ema_force_cpu else None)

            self.valid_f2ciw_ema = F2CIW()
            self.valid_f1normal_ema = F1Normal()

        self.all_time_f2ciw = 0.
        self.all_time_f1normal = 0.

    def ema_forward(self, x):
        feat_map = self.backbone_ema.module(x)
        logits = self.head_ema.module(feat_map)
        return logits

    def forward(self, x):
        feat_map = self.backbone(x)
        logits = self.head(feat_map)
        return logits

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)

        if not self.autocast_loss:
            logits = logits.float()

        with torch.cuda.amp.autocast(enabled=self.autocast_loss):
            total_loss = self.criterion(logits, y)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        logits = self(x)
        if not self.autocast_loss:
            logits = logits.float()
        with torch.cuda.amp.autocast(enabled=self.autocast_loss):
            total_loss = self.validation_criterion(logits, y)

        self.log('val_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.valid_f2ciw(logits.sigmoid(), y)
        self.valid_f1normal(logits.sigmoid(), y)
        self.log('val_F2CIW', self.valid_f2ciw, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_F1Normal', self.valid_f1normal, on_step=False, on_epoch=True, prog_bar=False)

        if self.hparams.args.model_ema and not self.hparams.args.model_ema_force_cpu:
            logits_ema = self.ema_forward(x)
            if not self.autocast_loss:
                logits_ema = logits_ema.float()
            with torch.cuda.amp.autocast(enabled=self.autocast_loss):
                total_loss_ema = self.validation_criterion(logits_ema, y)

            self.log('val_loss_ema', total_loss_ema, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

            self.valid_f2ciw_ema(logits_ema.sigmoid(), y)
            self.valid_f1normal_ema(logits_ema.sigmoid(), y)
            self.log('val_F2CIW_ema', self.valid_f2ciw_ema, on_step=False, on_epoch=True, prog_bar=False)
            self.log('val_F1Normal_ema', self.valid_f1normal_ema, on_step=False, on_epoch=True, prog_bar=False)

        return total_loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.hparams.args.model_ema:
            self.backbone_ema.update(self.backbone)
            self.head_ema.update(self.head)

    def on_after_backward(self, *args, **kwargs):
        if self.trainer.global_step < self.hparams.args.freeze_cluster_niters and self.sinkhorn_head:
            self.head.reset_tokenizer_grad()

    def validation_epoch_end(self, *args, **kwargs):

        if self.hparams.args.model_ema and not self.hparams.args.model_ema_force_cpu:
            max_f2ciw = max(self.trainer.logged_metrics["val_F2CIW"], self.trainer.logged_metrics["val_F2CIW_ema"])
            max_f1normal = max(self.trainer.logged_metrics["val_F1Normal"],
                               self.trainer.logged_metrics["val_F1Normal_ema"])
        else:
            max_f2ciw = self.trainer.logged_metrics["val_F2CIW"]
            max_f1normal = self.trainer.logged_metrics["val_F1Normal"]
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
            for idx, child in enumerate(self.backbone.backbone.children()):
                if idx <= self.backbone.freeze_index[self.hparams.args.freeze_layer_index]:
                    for param in child.parameters():
                        param.requires_grad = False

        params_backbone = optimizer.adjusted_parameter_setting(self.backbone, self.hparams.args.lr,
                                                               self.hparams.args.weight_decay)
        params_head = optimizer.adjusted_parameter_setting(self.head, self.hparams.args.lr,
                                                           self.hparams.args.weight_decay)
        params = params_backbone + params_head

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

    train_transform = transforms.create_sewerml_train_transformations(
        {"img_size": args.img_size, "model_name": args.backbone_model})
    eval_transform = transforms.create_sewerml_eval_transformations(
        {"img_size": args.img_size, "model_name": args.backbone_model})

    dm = datamodules.get_datamodule(dataset=args.dataset, batch_size=args.batch_size, workers=args.workers,
                                    ann_root=args.ann_root, data_root=args.data_root, train_transform=train_transform,
                                    eval_transform=eval_transform)

    dm.prepare_data()
    dm.setup("fit")

    weights = class_weight.effective_samples(dm.train_dataset.labels, dm.num_classes, args.effective_beta)
    model = Model(num_classes=dm.num_classes, loss_weight=weights, sigmoid_loss=args.sigmoid_loss, args=args)

    # Setup Logger
    version = "version_" + str(args.log_version)
    model_name = args.dataset + "_" + args.backbone_model + "_" + args.head_model + "_" + args.block_type + "_" + args.tokenizer_layer + "_" + version
    print("-" * 15 + model_name + "-" * 15)

    os.makedirs(args.log_save_dir, exist_ok=True)

    logger = WandbLogger(project=args.wandb_project,  # group runs in "MNIST" project
                         log_model='all',
                         save_dir=args.log_save_dir,
                         version=version,
                         name=model_name,
                         **{"group": args.wandb_group})  # log all new checkpoints during training

    logger_path = os.path.join(args.log_save_dir, model_name, "version_" + str(args.log_version))
    if args.monitor_metric:
        monitor = "valid_max_F2CIW"
        filename = '{epoch:02d}-{valid_max_F2CIW:.4f}'
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
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=64, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--log_save_dir', type=str, default="")
    parser.add_argument('--log_version', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataset', type=str, choices=["SewerML"], default="SewerML")
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--wandb_group', type=str, default="")
    parser.add_argument('--monitor_metric', action='store_true')
    parser.add_argument('--save_last', action='store_true')
    parser.add_argument('--save_top_k', type=int, default=1)

    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    run_cli()