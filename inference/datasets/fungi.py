import os

from .utils import Datum, DatasetBase

template = ['a photo of a {}, a type of mushroom.']


class Fungi(DatasetBase):

    dataset_dir = 'fungi'

    def __init__(self, root, num_shots):
        self.root = root
        self.template = template
        train = self.get_split('train')
        val = self.get_split('val')
        test = self.get_split('test')
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        super().__init__(train_x=train, val=val, test=test)
    
    def get_split(self, split='train'):
        items = []
        cnames = os.listdir(f"{self.root}/{self.dataset_dir}/{split}")
        for i, cname in enumerate(cnames):
            cname_text = ' '.join(cname.split('_')[1:])
            for f in os.listdir(f"{self.root}/{self.dataset_dir}/{split}/{cname}"):
                item = Datum(
                    impath=f"{self.root}/{self.dataset_dir}/{split}/{cname}/{f}",
                    label=i,
                    classname=cname_text
                )
                items.append(item)
        return items