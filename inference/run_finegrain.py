import os
import random
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import logging
logging.basicConfig(level = logging.INFO)

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
from methods.zs_clip import ZeroShotCLIP
from methods.tip_adaptor import TipAdaptor, TipAdaptorF
from methods.diffusion_classifier import DiffusionClassifier
from methods.dreambooth import Dreambooth
from methods.combined_lora import CombinedLora
import tensorboard
import re


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='fgvc', help='settings in yaml format. store in ./configs/METHOD/CONFIG.yaml')
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--method', type=str, default='clip', help='clip | diffusionclf | dreambooth | combinedlora')
    parser.add_argument('--log_root', type=str, default='./logs')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--device', type=int, default=0, help='>=0 for which gpu, <0 for all gpus')
    parser.add_argument('--seed', type=int, default=1, help='random seed value')

    # method specific configs
    parser.add_argument('--sd', type=str, default='2.0')
    parser.add_argument('--lora_dir', type=str, default='')
    parser.add_argument('--lora_iteration', type=int, default=0)
    parser.add_argument('--lora_token', type=str, default=None)
    parser.add_argument('--add_class_name', action="store_true")    # dreambooth only
    parser.add_argument('--unet_lora_filter', type=str, default='none', help="none|down/mirror/all-0/1/2, all-0 same as no filtering, down-2 is most extreme filtering")
    parser.add_argument('--test_weight_t', type=str, default='none')
    args = parser.parse_args()
    return args


def main():
    # Load config file
    args = get_arguments()
    args.device = f"cuda:{args.device}" if args.device >= 0 else "cuda"

    config_dir = f"./configs/{args.method}/{args.config}.yaml"
    assert (os.path.exists(config_dir))
    cfg = yaml.load(open(config_dir, 'r'), Loader=yaml.Loader)
    cfg['weight_t'] = args.test_weight_t
    
    # method specific config
    lora_methods = ['dreambooth', 'combinedlora']
    if args.method in lora_methods:
        cfg['version'] = '-'.join(args.sd.split('.'))       # overwrite
        cfg['lora_dir'] = args.lora_dir
        cfg['iteration'] = args.lora_iteration
        cfg['lora_token'] = args.lora_token
        cfg['add_class_name'] = args.add_class_name
        cfg['unet_lora_filter'] = args.unet_lora_filter
        
        suffix = f"_{args.config.split('_')[1]}" if len(args.config.split('_')) > 1 else ""
        lora_exp_name = os.path.basename(args.lora_dir)
        exp_name = f"{args.method}{suffix}_{lora_exp_name}_it{args.lora_iteration}"
        if cfg['weight_t'] != 'none':
            exp_name += f"_{cfg['weight_t']}"

        # extract configs from lora exp name
        match = re.search('_T(\d+)_', lora_exp_name)
        if match:
            cutoff_T = int(match.group(1))
        else:
            cutoff_T = -1
        cfg['cutoff_T'] = cutoff_T

        match = re.search('_r(\d+)_', lora_exp_name)
        if match:
            cfg['lora_rank'] = int(match.group(1))
        else:
            cfg['lora_rank'] = 8
    else:
        zero_shot_methods = ['clip', 'diffusionclf']
        if args.method in zero_shot_methods:
            exp_name = f"{args.method}_{args.config}"
        else:
            exp_name = f"{args.method}_{args.config}_{args.shots}_s{args.seed}"

    cache_dir = os.path.join(args.log_root, args.method, exp_name)
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['exp_name'] = exp_name
    cfg['log_root'] = args.log_root
    cfg['cfg_name'] = args.config
    cfg['shots'] = args.shots
    cfg['seed'] = args.seed

    # construct logger
    logger = logging.getLogger(exp_name)

    file_log_handler = logging.FileHandler(os.path.join(cache_dir, "log.txt"))
    logger.addHandler(file_log_handler)

    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)

    logger.info(f"Running experiments {exp_name}.")

    # Prepare data
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], args.data_root, args.shots, args.seed)

    # load cached if exists; otherwise save train_x (few-shot train set)
    if dataset.load_if_exists(cache_dir, "train_x"):
        logger.info("Loaded cached few-shot train set.")
    else:
        logger.info("Cached new few-shot train set.")
        dataset.save(cache_dir, "train_x")
    
    # Prepare method
    if args.method == 'clip':
        method = ZeroShotCLIP(cfg, dataset, args.device)
    elif args.method == 'tip':
        method = TipAdaptor(cfg, dataset, args.device)
    elif args.method == 'tipf':
        method = TipAdaptorF(cfg, dataset, args.device)
    elif args.method == 'diffusionclf':
        method = DiffusionClassifier(cfg, dataset, args.device)
    elif args.method == 'dreambooth':
        method = Dreambooth(cfg, dataset, args.device)
    elif args.method == 'combinedlora':
        method = CombinedLora(cfg, dataset, args.device)
    else:
        raise NotImplemented(f'{args.method} not implemented!')
    
    best_acc = 0
    for i in range(cfg['train_epochs']):
        method.train_mode()
        method.train_one_epoch(i)
        method.eval_mode()
        val_acc = method.eval(split='val')
        if val_acc > best_acc:
            logger.info(f"New best acc {val_acc} at epoch {i}.")
            method.save('best')

    method.load('best')
    method.eval_mode()
    test_acc = method.eval(split='test')
    logger.info(f"Experiment {exp_name} test acc {test_acc}.")

if __name__ == '__main__':
    main()
