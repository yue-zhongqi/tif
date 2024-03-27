import os


stats = {
    'fgvc': {
        1: {'steps': 150, 'f': 30, 's':60},
        2: {'steps': 300, 'f': 50, 's':100},
        4: {'steps': 600, 'f': 100, 's':200},
        8: {'steps': 900, 'f': 200, 's':300},
        16: {'steps': 1000, 'f': 200, 's':400},
    },
    'inat_Arachnida': {
        1: {'steps': 210, 'f': 30},
        2: {'steps': 200, 'f': 50},
        4: {'steps': 400, 'f': 50},
        8: {'steps': 1000, 'f': 200},
        16: {'steps': 1600, 'f': 200},
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
    'fungi': {
        1: {'steps': 200, 'f': 25, 's': 75},
        2: {'steps': 400, 'f': 50, 's': 150},
        4: {'steps': 700, 'f': 50, 's': 300},
        8: {'steps': 1000, 'f': 100, 's': 600},
        16: {'steps': 2200, 'f': 200, 's': 1000},
    },
    'isic2019': {
        1: {'steps': 300, 'f': 50, 's': 50},
        2: {'steps': 350, 'f': 50, 's': 100},
        4: {'steps': 600, 'f': 100, 's': 200},
        8: {'steps': 1200, 'f': 100, 's': 400},
        16: {'steps': 1600, 'f': 100, 's': 400},
    },
}

def run_lora(dataset, shots, lora_rank=8, cutoff_T=-1,\
             training_steps=-1, save_freq=-1, sd='2.0', \
             num_class_images=100, seed=1, jittering=1, cc=False, \
             rare_token='hta', add_class_name=False, weight_t='none', \
             unet_lora_filter='none', priority='high'):
    if training_steps < 0:
        training_steps = stats[dataset][shots]['steps']
    if save_freq < 0:
        save_freq = stats[dataset][shots]['f']
    exp_name = f"{dataset}_{shots}_sd{sd}_r{lora_rank}"
    if num_class_images != 100:
        exp_name += f"_nc{num_class_images}"
    if cutoff_T >= 0:
        exp_name += f"_T{cutoff_T}"
    if add_class_name:
        exp_name += "_cname"
    if weight_t != 'none':
        exp_name += f"_w-{weight_t}"
    if rare_token != 'hta':
        exp_name += f"_t-{rare_token}"
    if unet_lora_filter != 'none':
        exp_name += f"_{unet_lora_filter}"
    if jittering != 1:
        exp_name += "_noj"
    if cc:
        exp_name += "_cc"
    exp_name += f"_s{seed}"

    save_root_path = f"./logs/dreambooth_train/lora/{exp_name}"
    # train_data_path = f"./splits/{dataset}/train_{shots}_{seed}.data"
    if not os.path.exists(save_root_path):
        print(f"{exp_name}: not started.")
        return
    
    s = save_freq if 's' not in stats[dataset][shots].keys() else stats[dataset][shots]['s']
    class_dirs = os.listdir(save_root_path)
    completed = 0
    incompleted = []
    for c in class_dirs:
        missing = 0
        class_lora_dir = os.path.join(save_root_path, c)
        for step in range(s, training_steps+1, save_freq):
            unet_dir = f"{class_lora_dir}/unet_{step}.pt"
            text_dir = f"{class_lora_dir}/text_{step}.pt"
            if os.path.exists(unet_dir) and os.path.exists(text_dir):
                continue
            else:
                missing += 1
        if missing == 0:
            completed += 1
        else:
            incompleted.append(missing)
    print(f"{exp_name}: completed {completed}, incomplete {len(incompleted)}.")

# # fgvc
run_lora('veri', 4, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=2)
for seed in [2,3]:
    run_lora('fgvc', 1, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', seed=seed)
    run_lora('fgvc', 2, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', seed=seed)
    run_lora('fgvc', 4, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', seed=seed)
    run_lora('fgvc', 8, lora_rank=8, add_class_name=True, unet_lora_filter='all-1', seed=seed)
    run_lora('fgvc', 16, lora_rank=8, add_class_name=True, unet_lora_filter='none', seed=seed)    

# ###### Ablation on filter location
# for filter in ['all-2','all-1','none']:
#     run_lora('duke', 1, lora_rank=16, add_class_name=False, unet_lora_filter=filter)
#     run_lora('duke', 2, lora_rank=16, add_class_name=False, unet_lora_filter=filter)
#     run_lora('duke', 4, lora_rank=16, add_class_name=False, unet_lora_filter=filter)
#     run_lora('duke', 8, lora_rank=16, add_class_name=False, unet_lora_filter=filter)
#     run_lora('duke', 16, lora_rank=16, add_class_name=False, unet_lora_filter=filter)

# ###### REFERENCE SET
# # # duke
# for seed in [2,3]:
#     run_lora('duke', 1, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('duke', 2, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('duke', 4, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('duke', 8, lora_rank=16, add_class_name=False, unet_lora_filter='all-1', seed=seed)
#     run_lora('duke', 16, lora_rank=16, add_class_name=False, unet_lora_filter='none', seed=seed)
# # veri
# for seed in [2,3]:
#     run_lora('veri', 1, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('veri', 2, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('veri', 4, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', seed=seed)
#     run_lora('veri', 8, lora_rank=16, add_class_name=False, unet_lora_filter='all-1', seed=seed)
#     run_lora('veri', 16, lora_rank=16, add_class_name=False, unet_lora_filter='none', seed=seed)
# # isic
# for seed in [2,3]:
#     run_lora('isic2019', 1, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', seed=seed)
#     run_lora('isic2019', 2, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', seed=seed)
#     run_lora('isic2019', 4, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', seed=seed)
#     run_lora('isic2019', 8, lora_rank=8, add_class_name=True, unet_lora_filter='all-1', seed=seed)
#     run_lora('isic2019', 16, lora_rank=8, add_class_name=True, unet_lora_filter='none', seed=seed)

# # # fgvc





# for r in [16]:
#     for filter in ['all-2','all-1','none']:
#         for name in [True, False]:
#             run_lora('fgvc', 1, lora_rank=r, add_class_name=name, unet_lora_filter=filter)
#             run_lora('fgvc', 2, lora_rank=r, add_class_name=name, unet_lora_filter=filter)
#             run_lora('fgvc', 4, lora_rank=r, add_class_name=name, unet_lora_filter=filter)
#             run_lora('fgvc', 8, lora_rank=r, add_class_name=name, unet_lora_filter=filter)
#             run_lora('fgvc', 16, lora_rank=r, add_class_name=name, unet_lora_filter=filter)

# for r in [8]:
#     for filter in ['all-2','all-1']:
#         for name in [True]:
#             run_lora('fgvc', 1, lora_rank=r, add_class_name=name, unet_lora_filter=filter)
#             run_lora('fgvc', 2, lora_rank=r, add_class_name=name, unet_lora_filter=filter)
#             run_lora('fgvc', 4, lora_rank=r, add_class_name=name, unet_lora_filter=filter)
            
#run_lora('fungi', 1, lora_rank=32, add_class_name=True, unet_lora_filter='all-1', cc=True)
#run_lora('fungi', 2, lora_rank=32, add_class_name=True, unet_lora_filter='all-1', cc=True)
#run_lora('fungi', 4, lora_rank=32, add_class_name=True, unet_lora_filter='all-1', cc=True)
#run_lora('fungi', 4, lora_rank=32, add_class_name=True, cc=True)
#run_lora('fungi', 8, lora_rank=32, add_class_name=True, cc=True)
#run_lora('fungi', 16, lora_rank=32, add_class_name=True, cc=True)

#rank = [16, 32]
#filters = ['all-1', 'none']
#for r in rank:
#    for filter in filters:
#        for jitter in [0,1]:
#            run_lora('duke', 1, lora_rank=r, add_class_name=False, unet_lora_filter=filter, jittering=jitter)
#            run_lora('duke', 2, lora_rank=r, add_class_name=False, unet_lora_filter=filter, jittering=jitter)
#            run_lora('duke', 4, lora_rank=r, add_class_name=False, unet_lora_filter=filter, jittering=jitter)


#run_lora('duke', 1, lora_rank=32, add_class_name=False, unet_lora_filter='none')
#run_lora('duke', 1, lora_rank=32, add_class_name=False, unet_lora_filter='none', jittering=0)
#run_lora('duke', 2, lora_rank=32, add_class_name=False, unet_lora_filter='none')
#run_lora('duke', 4, lora_rank=32, add_class_name=False, unet_lora_filter='none')
#run_lora('duke', 4, lora_rank=32, add_class_name=False, unet_lora_filter='none', jittering=0)
#run_lora('duke', 8, lora_rank=32, add_class_name=False, unet_lora_filter='none')
#run_lora('duke', 16, lora_rank=32, add_class_name=False, unet_lora_filter='none')
#run_lora('duke', 8, lora_rank=64, add_class_name=False, unet_lora_filter='none')
#run_lora('duke', 16, lora_rank=64, add_class_name=False, unet_lora_filter='none')