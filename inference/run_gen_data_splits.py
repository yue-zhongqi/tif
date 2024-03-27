import os
import random
import torch
import numpy as np
from datasets import build_dataset
from torchvision.datasets import INaturalist
from PIL import Image


def main(dataset_name, data_root, log_root, shots, seed):
    # Prepare data
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = build_dataset(dataset_name, data_root, shots, seed=seed)
    save_root = f"{log_root}/data_splits/{dataset_name}"
    os.makedirs(save_root, exist_ok=True)

    def save_split(split='train'):
        if split == 'train':
            fname = f"train_{shots}_{seed}.data"
            split_data = dataset.train_x
        else:
            fname = f"{split}.data"
            split_data = getattr(dataset, split)
        if not os.path.exists(f"{save_root}/{fname}"):
            print(f"saving {dataset_name} split {split}...")
            dict_data = [{"classname": x.classname, "impath": x.impath, "label": x.label} for x in split_data]
            torch.save(dict_data, f"{save_root}/{fname}")
        return True
        
    save_split("train")
    save_split("val")
    save_split("test")


def gen_inat_split(data_root, log_root, super_category, shots=[1,2,4,8,16], seeds=[1,2,3]):
    train_raw = INaturalist(os.path.join(data_root, 'inat_train_mini'), version='2021_train_mini', target_type='full', download=False)
    val_raw = INaturalist(os.path.join(data_root, 'inat_val'), version='2021_valid', target_type='full', download=False)

    split_data_dir = f"{data_root}/inat_{super_category}"
    os.makedirs(f"{split_data_dir}/train", exist_ok=True)
    os.makedirs(f"{split_data_dir}/val", exist_ok=True)
    
    classes = [c for c in range(len(train_raw.all_categories)) if super_category in train_raw.all_categories[c]]
    # label_to_cidx = {l:classes[l] for l in range(len(classes))}
    # l_to_class_names = [' '.join(train_raw.all_categories[label_to_cidx[l]].split('_')[-2:]) for l in range(len(classes))]
    c_to_class_names = {c:' '.join(train_raw.all_categories[c].split('_')[-2:]) for c in classes}
    train_subset = [i for i in range(len(train_raw.index)) if train_raw.index[i][0] in classes]
    train_classes = [train_raw.index[i][0] for i in train_subset]
    val_subset = [i for i in range(len(val_raw.index)) if val_raw.index[i][0] in classes]
    used_indices = []
    save_root = f"{log_root}/data_splits/inat_{super_category}"
    os.makedirs(save_root, exist_ok=True)
    for shot in shots:
        for seed in seeds:
            np.random.seed(seed)
            splits_indices = []
            for c in classes:
                splits_indices += (np.random.choice(np.array(train_subset)[np.array(train_classes)==c], shot, replace=False)).tolist()
            assert len(set(splits_indices)) == len(splits_indices)
            assert len(splits_indices) == shot * len(classes)
            used_indices += splits_indices
            fname = f"train_{shot}_{seed}.data"
            # assert not os.path.exists(f"{save_root}/{fname}")
            dict_data = [{
                "classname": c_to_class_names[train_raw.index[idx][0]],
                "impath": f"{split_data_dir}/train/{idx}.png",
                "label": classes.index(train_raw.index[idx][0])
            } for idx in splits_indices]
            torch.save(dict_data, f"{save_root}/{fname}")
            torch.save(dict_data, f"{split_data_dir}/{fname}")

    for idx in set(used_indices):
        cat_id, fname = train_raw.index[idx]
        img = Image.open(os.path.join(train_raw.root, train_raw.all_categories[cat_id], fname))
        img.save(f"{split_data_dir}/train/{idx}.png")

    dict_data = []
    for idx in val_subset:
        cat_id, fname = val_raw.index[idx]
        img = Image.open(os.path.join(val_raw.root, val_raw.all_categories[cat_id], fname))
        img.save(f"{split_data_dir}/val/{idx}.png")
        dict_data.append({
            "classname": c_to_class_names[cat_id],
            "impath": f"{split_data_dir}/val/{idx}.png",
            "label": classes.index(cat_id)
        })
    torch.save(dict_data, f"{save_root}/val.data")
    torch.save(dict_data, f"{split_data_dir}/val.data")

for dataset_name in ['isic2019']:
    for shots in [1,2,4,8,16]:
        for seed in [1,2,3]:
            main(dataset_name, './data', './logs', shots, seed)

#for dataset_name in ['fgvc', 'cub', 'stanford_cars', 'dtd', 'eurosat']:
#    for shots in [1,2,4,8,16]:
#        for seed in [1,2,3]:
#            main(dataset_name, './data', './logs', shots, seed)

# generate inat split
#'Arachnida'
#gen_inat_split('./data', './logs', 'Arachnida')