import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from collections import Counter
import torch

template = ['a photo of car {} captured by surveillance camera.']


class Veri(DatasetBase):

    dataset_dir = 'veri'

    def __init__(self, root, num_shots):
        self.root = root
        self.template = template

        gallery_files = os.listdir(f"{root}/{self.dataset_dir}/image_test")
        query_files = os.listdir(f"{root}/{self.dataset_dir}/image_query")

        gallery_ids = [file[:4] for file in gallery_files]
        self.unique_ids = list(set(gallery_ids))

        train = self.get_datums(gallery_files, 'image_test')
        test = self.get_datums(query_files, 'image_query')
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        val = train
        super().__init__(train_x=train, val=val, test=test)

    def get_datums(self, files, folder_name):
        items = []
        for f in files:
            id = f[:4]
            item = Datum(
                impath=f"{self.root}/{self.dataset_dir}/{folder_name}/{f}",
                label=self.unique_ids.index(id),
                classname=f"car {int(id)}"
            )
            items.append(item)
        return items