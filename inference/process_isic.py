from PIL import Image, ImageOps
from collections import Counter
import os
import csv
import torch
import random


dataset = './data/ISIC_2019_Training_Input'
target_dataset = './data/isic2019'
gt_file = '0_gt.csv'

rows = []
with open(f'{dataset}/{gt_file}', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        rows.append(row)

print(rows[0])

classnames = ["Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma", "Actinic Keratosis", "Benign Keratosis-like Lesions", "Dermatofibroma", "Vascular Lesion", "Squamous Cell Carcinoma"]

all_data = {}
for c in range(len(classnames)):
    all_data[c] = []

for row in rows[1:]:
    file_name = f"{row[0]}.jpg"
    label = row[1:].index('1.0')
    all_data[label].append(file_name)

for c in range(len(classnames)):
    print(len(all_data[c]))

# generate train, val and test split
p_val = 0.1
p_test = 0.5
train_data = {}
val_data = {}
test_data = {}

for c in range(len(classnames)):
    c_data = all_data[c]
    n_val = int(len(c_data) * p_val)
    n_test = int(len(c_data) * p_test)
    val = random.sample(c_data, n_val)
    left = [x for x in c_data if x not in val]
    test = random.sample(left, n_test)
    train = [x for x in left if x not in test]
    train_data[c] = train
    val_data[c] = val
    test_data[c] = test


def resize_and_center_crop(image, target_size=(512, 512)):
    # Calculate the target aspect ratio
    target_width, target_height = target_size
    width, height = image.size

    # Determine which side is the limiting factor
    if width / height < target_width / target_height:
        # Width is the limiting factor, so scale height
        new_height = int(height * target_width / width)
        new_width = target_width
    else:
        # Height is the limiting factor, so scale width
        new_width = int(width * target_height / height)
        new_height = target_height

    # Resize the image
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Calculate coordinates for cropping
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2

    # Crop the center
    image = image.crop((left, top, right, bottom))
    return image


def process_data(data, split='train'):
    for c in range(len(classnames)):
        c_data = data[c]
        cname = classnames[c]
        target_dir = f"{target_dataset}/{split}/{c}_{cname.replace(' ', '_')}"
        os.makedirs(target_dir, exist_ok=True)
        for idx, file in enumerate(c_data):
            original_image = Image.open(f"{dataset}/{file}")
            cropped_image = resize_and_center_crop(original_image)
            cropped_image.save(f"{target_dir}/{idx}.jpg")
            print(f"Processed {split} class {c}/{len(classnames)}, {idx}/{len(c_data)}...")

process_data(train_data, 'train')
process_data(val_data, 'val')
process_data(test_data, 'test')
