import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from collections import Counter
import torch




class ISIC2019(DatasetBase):

    dataset_dir = 'isic2019'
    classnames = ["Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma", "Actinic Keratosis", "Benign Keratosis-like Lesions", "Dermatofibroma", "Vascular Lesion", "Squamous Cell Carcinoma"]
    template = ['a high quality dermoscopic image of {}.']
    max_test_per_class = 600
    def __init__(self, root, num_shots):
        self.root = root
        train = self.process_split('train')
        val = self.process_split('val')
        test = self.process_split('test')
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        super().__init__(train_x=train, val=val, test=test)

    def process_split(self, split='train'):
        cnames = os.listdir(f"{self.root}/{self.dataset_dir}/{split}")
        items = []
        for cname in cnames:
            c_dir = f"{self.root}/{self.dataset_dir}/{split}/{cname}"
            label = int(cname.split('_')[0])
            files = os.listdir(c_dir)
            if len(files) > self.max_test_per_class and split == 'test':
                files = files[:self.max_test_per_class]
            for f in files:
                item = Datum(
                    impath=f"{c_dir}/{f}",
                    label=label,
                    classname=self.classnames[label]
                )
                items.append(item)
        return items