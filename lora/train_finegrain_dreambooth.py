import os
import shutil
import torch
import argparse

######## User settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./train_logs')
parser.add_argument('--dataset', type=str, default='fgvc', help='dataset name')
parser.add_argument('--shots', type=str, default='16', help='how many shots')
parser.add_argument('--prompt_template', type=str, default='a photo of a {}, a type of aircraft.')
parser.add_argument('--class_prompt', type=str, default='a photo of an aircraft.')

parser.add_argument('--training_steps', type=int, default=1000)
parser.add_argument('--save_start', type=int, default=0)
parser.add_argument('--save_freq', type=int, default=200)
parser.add_argument('--sd', type=str, default='2.0')
parser.add_argument('--num_class_images', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)

# HYPERPARAMETERS
parser.add_argument('--rare_token', type=str, default='hta')    # if =='none', use no rare token
parser.add_argument('--add_class_name', action="store_true")        # True: a photo of a hta 737/800, a type of airplane
parser.add_argument('--lora_rank', type=int, default=8)
parser.add_argument('--cutoff_T', type=int, default=-1)
parser.add_argument('--jittering', type=int, default=1)
parser.add_argument('--center_crop', action="store_true")
parser.add_argument('--weight_t', type=str, default='none', help="none | pdae")
parser.add_argument('--unet_lora_filter', type=str, default='none', help='none | down/mirror/all-0/1/2')

args = parser.parse_args()
#################################
assert args.rare_token != 'none' or args.add_class_name

dataset = args.dataset
shots = args.shots
class_prompt = args.class_prompt
lora_rank = args.lora_rank
training_steps = args.training_steps
sd_version = args.sd
num_class_images = args.num_class_images
seed = args.seed
save_freq = args.save_freq

# fixed settings
class_data_path = f"{args.root}/dreambooth_train/prior_data/{'_'.join(class_prompt.split(' '))}_{sd_version}"
train_data_path = f"./splits/{dataset}/train_{shots}_{seed}.data"

exp_name = f"{dataset}_{shots}_sd{sd_version}_r{lora_rank}"
if num_class_images != 100:
    exp_name += f"_nc{num_class_images}"
if args.cutoff_T >= 0:
    exp_name += f"_T{args.cutoff_T}"
if args.add_class_name:
    exp_name += "_cname"
if args.weight_t != 'none':
    exp_name += f"_w-{args.weight_t}"
if args.rare_token != 'hta':
    exp_name += f"_t-{args.rare_token}"
if args.unet_lora_filter != 'none':
    exp_name += f"_{args.unet_lora_filter}"
if args.jittering != 1:
    exp_name += "_noj"
if args.center_crop:
    exp_name += "_cc"
exp_name += f"_s{seed}"

save_root_path = f"{args.root}/dreambooth_train/lora/{exp_name}"
tmp_dataset_folder = f"{args.root}/dreambooth_train/tmp/{exp_name}"

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
all_data = torch.load(train_data_path)
class_names = list(set([data["classname"] for data in all_data]))
print(class_names)
for class_name in class_names:
    class_name_dir = class_name.replace('_', ' ')
    class_name_dir = class_name_dir.replace("/", "-")
    class_lora_dir = f"{save_root_path}/{'_'.join(class_name_dir.split(' '))}"

    instance_prompt = args.prompt_template.format(args.rare_token) if not args.add_class_name else \
                        args.prompt_template.format(f"{args.rare_token} {class_name}")
    if args.rare_token == 'none':
        instance_prompt = args.prompt_template.format(class_name)

    max_missing_step = 0
    s = max(save_freq, args.save_start)
    for step in range(s, training_steps+1, save_freq):
        unet_dir = f"{class_lora_dir}/unet_{step}.pt"
        text_dir = f"{class_lora_dir}/text_{step}.pt"
        if os.path.exists(unet_dir) and os.path.exists(text_dir):
            continue
        else:
            max_missing_step = step

    if max_missing_step == 0:
        print(f"{class_name} LoRAs exists! Skipping...")
        continue
    else:
        # Proceed to training
        print(f"{class_name} LoRAs missing step till {max_missing_step}. Training...")
        os.makedirs(class_lora_dir, exist_ok=True)
        if os.path.exists(tmp_dataset_folder):
            shutil.rmtree(tmp_dataset_folder)
        os.makedirs(tmp_dataset_folder)

        # copy train images to tmp folder
        for data in all_data:
            if data["classname"] == class_name:
                os.system(f"cp {data['impath']} {tmp_dataset_folder}")

        # run training
        print(f"Training {class_name} LoRA...")
        print(f"Instance prompt {instance_prompt}")
        print(f"Class prompt {class_prompt}")
        print(f"Exp name {exp_name}")

        cmd = f"accelerate launch --mixed_precision fp16 \
            train_lora_dreambooth.py \
            --pretrained_model_name_or_path={MODEL_IDS[sd_version]}  \
            --instance_data_dir={tmp_dataset_folder} \
            --output_dir={class_lora_dir} \
            --instance_prompt=\"{instance_prompt}\" \
            --train_text_encoder \
            --resolution=512 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=1 \
            --learning_rate=1e-4 \
            --learning_rate_text=5e-5 \
            --lr_scheduler=constant \
            --lr_warmup_steps=0 \
            --max_train_steps={max_missing_step} \
            --save_steps={save_freq} \
            --save_start={args.save_start} \
            --with_prior_preservation \
            --num_class_images {num_class_images} \
            --lora_rank={lora_rank} \
            --cutoff_T={args.cutoff_T} \
            --weight_t={args.weight_t} \
            --unet_lora_filter={args.unet_lora_filter} \
            --class_data_dir={class_data_path} \
            --class_prompt=\"{class_prompt}\""
        if args.jittering > 0:
            cmd += "  --color_jitter"
        if args.center_crop:
            cmd += " --center_crop"
        os.system(cmd)
        
#  python train_finegrain_dreambooth.py --root=./logs --dataset=duke --shots=8 --prompt_template='a photo of {} captured by surveillance camera' --class_prompt='a person captured by surveillance camera' --training_steps=800 --save_freq=100 --lora_rank=16