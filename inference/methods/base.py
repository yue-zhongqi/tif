from abc import ABC, abstractmethod
import logging


class BaseMethod(ABC):
    def __init__(self, config, dataset, device):
        self.cfg = config
        self.device = device
        self.dataset = dataset
        self.logger = logging.getLogger(config["exp_name"])

    @abstractmethod
    def train_mode(self):
        pass

    @abstractmethod
    def eval_mode(self):
        pass
    
    @abstractmethod
    def train_one_epoch(self):
        pass

    @abstractmethod
    def eval(self, split):
        pass

    @abstractmethod
    def save(self, name):
        pass

    @abstractmethod
    def load(self, name):
        pass


