import os
import shutil
import torch
import argparse

######## User settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='fgvc', help='dataset name')
parser.add_argument('--root', type=str, default='./logs', help='root directory for logging/saving models.')
parser.add_argument('--prompt_template', type=str, default='a photo of {}, a type of airplane')
parser.add_argument('--class_prompt', type=str, default='a photo of airplane')
parser.add_argument('--shots', type=str, default='16', help='how many shots')

parser.add_argument('--training_steps', type=int, default=36000)
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--sd', type=str, default='2.0')
parser.add_argument('--num_class_images', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)

# HYPERPARAMETERS
parser.add_argument('--instance_prompt', type=str, default=None)
parser.add_argument('--lora_rank', type=int, default=32)
parser.add_argument('--cutoff_T', type=int, default=-1)
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()
#################################

dataset = args.dataset
shots = args.shots
instance_prompt = args.instance_prompt
class_prompt = args.class_prompt
lora_rank = args.lora_rank
training_steps = args.training_steps
sd_version = args.sd
num_class_images = args.num_class_images
seed = args.seed
save_freq = args.save_freq

if args.debug:
    training_steps = 20
    save_freq = 10
    num_class_images = 32

# fixed settings
exp = f"{dataset}_{shots}_sd{sd_version}_r{lora_rank}_s{seed}"
if args.cutoff_T >= 0:
    exp += f"_T{args.cutoff_T}"
if instance_prompt is not None:
    exp += f"_{instance_prompt}"

class_data_path = f"{args.root}/dreambooth_train/prior_data/{'_'.join(class_prompt.split(' '))}"
train_data_path = f"./splits/{dataset}/train_{shots}_{seed}.data"
save_root_path = f"{args.root}/dreambooth_train/combined_lora/{exp}"



# RUNNING EXPERIMENTS
MODEL_IDS = {
    '1.1': "CompVis/stable-diffusion-v1-1",
    '1.2': "CompVis/stable-diffusion-v1-2",
    '1.3': "CompVis/stable-diffusion-v1-3",
    '1.4': "CompVis/stable-diffusion-v1-4",
    '1.5': "runwayml/stable-diffusion-v1-5",
    '2.0': "stabilityai/stable-diffusion-2-base",
    '2.1': "stabilityai/stable-diffusion-2-1-base"
}
cmd = f"accelerate launch --mixed_precision fp16 \
    train_lora_combined.py \
    --pretrained_model_name_or_path={MODEL_IDS[sd_version]}  \
    --instance_data_dir={train_data_path} \
    --output_dir={save_root_path} \
    --prompt_template=\"{args.prompt_template}\" \
    --train_text_encoder \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --learning_rate_text=5e-5 \
    --color_jitter \
    --lr_scheduler=constant \
    --lr_warmup_steps=0 \
    --max_train_steps={training_steps} \
    --save_steps={save_freq} \
    --with_prior_preservation \
    --num_class_images {num_class_images} \
    --lora_rank={lora_rank} \
    --class_data_dir={class_data_path} \
    --class_prompt=\"{class_prompt}\" \
    --cutoff_T {args.cutoff_T}"
if instance_prompt is not None:
    cmd += f" --instance_prompt={instance_prompt}"
os.system(cmd)

# python train_finegrain_combined.py --dataset cub --shots 4 --prompt_template "a photo of {}, a type of bird" --class_prompt "a photo of bird" --lora_rank 64 --training_steps 50000 --save_freq 1000 --seed 1