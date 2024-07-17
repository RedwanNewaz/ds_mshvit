import subprocess
import yaml
from time import sleep
from pprint import pprint
import re
import os
from subprocess import Popen, PIPE, CalledProcessError
from argparse import ArgumentParser

defaultParams = [ "--precision", "16", "--max_epochs", "40", "--gpus", "1", "--img_size", "224", "--deterministic",
    "--transformer_depth", "2", "--block_drop", "0.", "--tokenizer_layer", "Sinkhorn", "--block_type", "FNetBlock", "--use_pos_embed", "--num_heads", "8",
    "--qkv_bias", "--mlp_ratio", "4.", "--proj_drop", "0.", "--attn_drop", "0.", "--pos_embed_drop", "0.", "--cross_num_heads", "8", "--cross_qkv_bias",
    "--cross_proj_drop", "0.", "--cross_attn_drop", "0.", "--cross_block_drop", "0.", "--num_cluster", "64", "--sinkhorn_eps", "0.25", "--sinkhorn_iters", "5", "--l2_normalize",
    "--backbone_feature_maps", "layer3", "layer4", "--shared_tower", "--multiscale_method", "CrossScale-Token", "--sigmoid_loss", "--monitor_metric",
    "--progress_bar_refresh_rate", "1000", "--flush_logs_every_n_steps", "1000", "--log_every_n_steps", "1"
                  ]

def jointTrainer(config):
    cmd = [config["python"], config["exe"],
            "--ann_root", config["annotation"],
           "--rgb_data_root", config["datasets"]["rgb"],
           "--opti_data_root", config["datasets"]["opti"],
           "--baseline_model_path", config["weights"]["rgbModel"],
           "--optical_model_path", config["weights"]["optiModel"],
           "--batch_size", config["batch_size"],
           "--wandb_project", config["wand"]["project"],
           "--wandb_group", config["wand"]["group"],
           "--log_save_dir", config["output"],
           "--backbone_model", config["model"]["backbone"],
           "--head_model", config["model"]["head"],
           "--token_dim",  config["token_dim"],
           "--log_version", config["log_version"],
           "--seed", config["seed"]
           ] + defaultParams
    cmd = list(map(str, cmd))
    pprint(cmd)
    with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')  # process line here

    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)

def isGPUAvailable(config):
    command = "nvidia-smi"
    while True:
        process = Popen([command], stdout=PIPE, shell=True)
        out_value = process.communicate()[0]
        out_value = str(out_value.decode("utf-8")).split("\n")

        monitorLines = [30, 31, 32, 33]
        for i, out in enumerate(out_value):
            if i in monitorLines:
                res = out.split(" ")
                res = list(filter(len, res))
                index = int(res[1])
                usage = int(re.findall("\d+", res[-2])[0])

                if usage <= 100:
                    print(i, out)
                    config["CUDA_VISIBLE_DEVICES"] = index
                    return True
                print(out)

        sleep(1)

    return False

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help='config yaml file')
    parser.add_argument("--module", type=str, default="joint_trainer1", help="specify training config")

    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.safe_load(file)


    # if isGPUAvailable(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{config["CUDA_VISIBLE_DEVICES"]}'
    jointTrainer(config[args.module])

