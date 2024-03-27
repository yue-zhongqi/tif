import os
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from torchvision.datasets import INaturalist
import torch
template = ['a photo of a {}, a type of aircraft.']


class iNat(DatasetBase):

    def __init__(self, root, super_category, num_shots, seed):
        train_file = os.path.join(root, f"inat_{super_category}", f"train_{num_shots}_{seed}.data")
        val_file = os.path.join(root, f"inat_{super_category}", f"val.data")
        assert os.path.exists(train_file)
        assert os.path.exists(val_file)
        train = self.read_data(train_file)
        val = self.read_data(val_file)
        self.template = ['a photo of a {}, a type of ' + super_category + '.']
        # train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        super().__init__(train_x=train, val=val, test=val)
    
    def read_data(self, file):
        items = torch.load(file)
        datums = []
        for item in items:
            datum = Datum(
                impath=item['impath'],
                label=item['label'],
                classname=item['classname']
            )
            datums.append(datum)
        return datums