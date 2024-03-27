import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader


template = ['a photo of a {}, a type of bird.']


class CUB(DatasetBase):

    dataset_dir = 'cub'

    def __init__(self, root, num_shots):
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.template = template

        train = self.read_data('train')
        val = self.read_data('test')
        test = self.read_data('test')
        
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, split):
        items = []
        split_dir = os.path.join(self.dataset_dir, split)
        class_names = os.listdir(split_dir)
        for i, name in enumerate(class_names):
            class_name = ' '.join(name[4:].split('_'))
            class_dir = os.path.join(split_dir, name)
            files = os.listdir(class_dir)
            for file in files:
                item = Datum(
                    impath=os.path.join(class_dir, file),
                    label=i,
                    classname=class_name
                )
                items.append(item)
        return items