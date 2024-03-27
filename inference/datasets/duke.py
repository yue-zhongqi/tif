import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from collections import Counter
import torch

template = ['a photo of {}.']


class Duke(DatasetBase):

    dataset_dir = 'duke'

    def __init__(self, root, num_shots):
        self.root = root
        self.template = template

        '''
        gallery_files = os.listdir(f"{root}/{self.dataset_dir}/bounding_box_test")
        gallery_ids = [file[:4] for file in gallery_files]
        unique_ids = list(set(gallery_ids))
        qualified_ids = [id for id in unique_ids if gallery_ids.count(id) >= 16]

        query_files = os.listdir(f"{root}/{self.dataset_dir}/query")
        query_ids = [file[:4] for file in query_files if file[:4] in qualified_ids]
        counter = Counter(query_ids)
        top_k_items = counter.most_common(150)

        self.qualified_ids = [item[0] for item in top_k_items]
        gallery_files = [f for f in gallery_files if f[:4] in self.qualified_ids]
        query_files = [f for f in query_files if f[:4] in self.qualified_ids]

        train = self.get_datums(gallery_files, 'bounding_box_test')
        test = self.get_datums(query_files, 'query')
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        val = train
        '''
        train = self.process(f'./logs/splits/duke/train_{num_shots}_1.data')
        val = self.process(f'./logs/splits/duke/val.data')
        test = self.process(f'./logs/splits/duke/test.data')
        super().__init__(train_x=train, val=val, test=test)
    
    def process(self, file):
        data = torch.load(file)
        items = []
        for d in data:
            item = Datum(
                impath=d['impath'],
                label=d['label'],
                classname=d['classname']
            )
            items.append(item)
        return items

    def get_datums(self, files, folder_name):
        items = []
        for f in files:
            id = f[:4]
            item = Datum(
                impath=f"{self.root}/{self.dataset_dir}/{folder_name}/{f}",
                label=self.qualified_ids.index(id),
                classname=f"person {int(id)}"
            )
            items.append(item)
        return items