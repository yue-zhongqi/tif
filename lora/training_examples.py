##########################
# To run the code



import os

templates = {
    'fgvc': 'a photo of a {}, a type of aircraft.',
    'duke': 'a photo of person {} captured by surveillance camera.',
    'veri': 'a photo of car {} captured by surveillance camera.',
    'isic2019': 'a high quality dermoscopic image of {}.'
}

class_prompts = {
    'fgvc': 'a photo of an aircraft',
    'duke': 'a photo of a person captured by surveillance camera.',
    'veri': 'a photo of a car captured by surveillance camera.',
    'isic2019': 'a high quality dermoscopic image.'
}

stats = {
    'fgvc': {
        1: {'steps': 150, 'f': 30, 's':60},
        2: {'steps': 300, 'f': 50, 's':100},
        4: {'steps': 600, 'f': 100, 's':200},
        8: {'steps': 900, 'f': 200, 's':300},
        16: {'steps': 1000, 'f': 200, 's':400},
    },
    'duke': {
        1: {'steps': 250, 'f': 50, 's': 50},
        2: {'steps': 300, 'f': 50, 's': 100},
        4: {'steps': 600, 'f': 100, 's': 200},
        8: {'steps': 1000, 'f': 100, 's': 400},
        16: {'steps': 1200, 'f': 200, 's': 400},
    },
    'veri': {
        1: {'steps': 250, 'f': 50, 's': 50},
        2: {'steps': 300, 'f': 50, 's': 100},
        4: {'steps': 600, 'f': 100, 's': 200},
        8: {'steps': 1000, 'f': 100, 's': 400},
        16: {'steps': 1200, 'f': 200, 's': 400},
    },
    'isic2019': {
        1: {'steps': 300, 'f': 50, 's': 50},
        2: {'steps': 350, 'f': 50, 's': 100},
        4: {'steps': 600, 'f': 100, 's': 200},
        8: {'steps': 800, 'f': 100, 's': 200},
        16: {'steps': 1600, 'f': 100, 's': 400},
    },
}


def run_lora(dataset, shots, lora_rank=8, cutoff_T=-1,\
             training_steps=-1, save_freq=-1, sd='2.0', \
             num_class_images=100, seed=1, jittering=1, cc=False, \
             rare_token='hta', add_class_name=False, weight_t='none', \
             unet_lora_filter='none', priority='low'):
    assert rare_token != 'none' or add_class_name
    if training_steps < 0:
        training_steps = stats[dataset][shots]['steps']
    if save_freq < 0:
        save_freq = stats[dataset][shots]['f']

    # root = "/workspace/train_logs"    # internal
    root = "LOGS_DIR"
    cmd = f"python train_finegrain_dreambooth.py --root {root} \
        --dataset {dataset} --shots {shots} --sd {sd} \
        --prompt_template \"{templates[dataset]}\" \
        --class_prompt \"{class_prompts[dataset]}\" \
        --lora_rank {lora_rank} --num_class_images {num_class_images} \
        --training_steps {training_steps} --save_freq {save_freq} \
        --seed {seed} --rare_token {rare_token} --weight_t {weight_t} \
        --unet_lora_filter {unet_lora_filter} --jittering {jittering}"
    if cutoff_T >0 :
        cmd += f" --cutoff_T {cutoff_T}"
    if add_class_name:
        cmd += f" --add_class_name"
    if 's' in stats[dataset][shots].keys():
        cmd += f" --save_start {stats[dataset][shots]['s']}"
    if cc:
        cmd += " --center_crop"
    print(f"Running {cmd}")
    os.system(cmd)

###### EXAMPLE SET
# # fgvc
# for seed in [1,2,3]:
#     run_lora('fgvc', 1, lora_rank=16, add_class_name=True, unet_lora_filter='all-2', seed=seed)
#     run_lora('fgvc', 2, lora_rank=16, add_class_name=True, unet_lora_filter='all-2', seed=seed)
#     run_lora('fgvc', 4, lora_rank=16, add_class_name=True, unet_lora_filter='all-2', seed=seed)
#     run_lora('fgvc', 8, lora_rank=16, add_class_name=True, unet_lora_filter='all-1', seed=seed)
#     run_lora('fgvc', 16, lora_rank=16, add_class_name=True, unet_lora_filter='none', seed=seed)    
# # duke
# for seed in [1,2,3]:
#     run_lora('duke', 1, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('duke', 2, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('duke', 4, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('duke', 8, lora_rank=16, add_class_name=False, unet_lora_filter='all-1', seed=seed)
#     run_lora('duke', 16, lora_rank=16, add_class_name=False, unet_lora_filter='none', seed=seed)
# # veri
# for seed in [1,2,3]:
#     run_lora('veri', 1, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('veri', 2, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('veri', 4, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('veri', 8, lora_rank=16, add_class_name=False, unet_lora_filter='all-1', seed=seed)
#     run_lora('veri', 16, lora_rank=16, add_class_name=False, unet_lora_filter='none', seed=seed)
# # isic
# for seed in [1,2,3]:
#     run_lora('isic2019', 1, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', seed=seed)
#     run_lora('isic2019', 2, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', seed=seed)
#     run_lora('isic2019', 4, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', seed=seed)
#     run_lora('isic2019', 8, lora_rank=8, add_class_name=True, unet_lora_filter='all-1', seed=seed)
#     run_lora('isic2019', 16, lora_rank=8, add_class_name=True, unet_lora_filter='none', seed=seed)