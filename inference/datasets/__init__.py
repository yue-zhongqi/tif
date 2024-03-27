from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .cub import CUB
from .inat import iNat
from .duke import Duke
from .fungi import Fungi
from .veri import Veri
from .isic2019 import ISIC2019

dataset_list = {
    "oxford_pets": OxfordPets,
    "eurosat": EuroSAT,
    "ucf101": UCF101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "dtd": DescribableTextures,
    "fgvc": FGVCAircraft,
    "food101": Food101,
    "oxford_flowers": OxfordFlowers,
    "stanford_cars": StanfordCars,
    "cub": CUB,
    "inat": iNat,
    "duke": Duke,
    "fungi": Fungi,
    "veri": Veri,
    "isic2019": ISIC2019,
}


def build_dataset(dataset, root_path, shots, seed):
    if 'inat' not in dataset:
        return dataset_list[dataset](root_path, shots)
    else:
        cname = dataset.split('_')[1]
        return dataset_list['inat'](root_path, cname, shots, seed)