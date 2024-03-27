import os

def test_lora(dataset, shots, lora_iteration, \
              lora_rank=64, cutoff_T=-1, rare_token='hta', sd='2.0', seed=1, \
              num_class_images=100, add_class_name=False, weight_t='none', \
              unet_lora_filter='none', jittering=1, cc=False, test_weight_t='none', \
              test_seed=None, config=None, probe=False, priority='low'):
    assert rare_token != 'none' or add_class_name
    # GET EXP NAME (COPIED FROM LORA PROJECT)
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

    if test_seed is None:
        test_seed = seed
    if config is None:
        config = dataset
    lora_dir = f"./logs/dreambooth_train/lora/{exp_name}"

    if probe:
        file = 'run_lora_probe.py'
        name = 'probe.yml'
    else:
        file = 'run_finegrain.py'
        name = 'train_finegrain.yml'
    cmd = f"python {file} --config {config} --shots {shots} --method dreambooth --sd {sd} \
        --seed {test_seed} --lora_dir {lora_dir} --lora_token {rare_token} --lora_iteration {lora_iteration} \
        --unet_lora_filter {unet_lora_filter} --test_weight_t {test_weight_t}"
    if add_class_name:
        cmd += " --add_class_name"
    print(f"Running {cmd}")

    os.system(cmd)


# for seed in [1, 2,3]:
    # ### FGVC
    # test_lora('fgvc', 1, lora_rank=16, add_class_name=True, unet_lora_filter='all-2', lora_iteration=60, test_weight_t='tif', seed=seed)
    # test_lora('fgvc', 2, lora_rank=16, add_class_name=True, unet_lora_filter='all-2', lora_iteration=100, test_weight_t='tif', seed=seed)
    # test_lora('fgvc', 4, lora_rank=16, add_class_name=True, unet_lora_filter='all-2', lora_iteration=200, test_weight_t='tif', seed=seed)
    # test_lora('fgvc', 8, lora_rank=16, add_class_name=True, unet_lora_filter='all-1', lora_iteration=700, test_weight_t='tif', seed=seed)
    # test_lora('fgvc', 16, lora_rank=16, add_class_name=True, unet_lora_filter='none', lora_iteration=1000, test_weight_t='tif', seed=seed)

    # ### ISIC
    # test_lora('isic2019', 1, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', lora_iteration=150, test_weight_t='tif', seed=seed)
    # test_lora('isic2019', 2, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', lora_iteration=200, test_weight_t='tif', seed=seed)
    # test_lora('isic2019', 4, lora_rank=8, add_class_name=True, unet_lora_filter='all-2', lora_iteration=500, test_weight_t='tif', seed=seed)
    # test_lora('isic2019', 8, lora_rank=8, add_class_name=True, unet_lora_filter='all-1', lora_iteration=700, test_weight_t='tif', seed=seed)
    # test_lora('isic2019', 16, lora_rank=8, add_class_name=True, unet_lora_filter='none', lora_iteration=800, test_weight_t='tif', seed=seed)

    # ### DUKE
    # test_lora('duke', 1, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', lora_iteration=200, test_weight_t='tif', config='duke_v2', seed=seed)
    # test_lora('duke', 2, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', lora_iteration=250, test_weight_t='tif', config='duke_v2', seed=seed)
    # test_lora('duke', 4, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', lora_iteration=400, test_weight_t='tif', config='duke_v2', seed=seed)
    # test_lora('duke', 8, lora_rank=16, add_class_name=False, unet_lora_filter='all-1', lora_iteration=700, test_weight_t='tif', config='duke_v2', seed=seed)
    # test_lora('duke', 16, lora_rank=16, add_class_name=False, unet_lora_filter='none', lora_iteration=1000, test_weight_t='tif', config='duke_v2', seed=seed)

    # ### VERI
    # test_lora('veri', 1, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', lora_iteration=150, test_weight_t='tif', seed=seed)
    # test_lora('veri', 2, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', lora_iteration=200, test_weight_t='tif', seed=seed)
    # test_lora('veri', 4, lora_rank=16, add_class_name=False, unet_lora_filter='all-2', lora_iteration=400, test_weight_t='tif', seed=seed)
    # test_lora('veri', 8, lora_rank=16, add_class_name=False, unet_lora_filter='all-1', lora_iteration=500, test_weight_t='tif', seed=seed)
    # test_lora('veri', 16, lora_rank=16, add_class_name=False, unet_lora_filter='none', lora_iteration=1000, test_weight_t='tif', seed=seed)