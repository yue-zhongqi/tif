import clip
import torchvision.transforms as transforms
from methods.utils import clip_classifier, pre_load_features, cls_acc, classwise_acc
from datasets.utils import build_data_loader
from .base import BaseMethod
import open_clip


class ZeroShotCLIP(BaseMethod):
    def __init__(self, config, dataset, device):
        super().__init__(config, dataset, device)
        #self.clip_model, preprocess = clip.load(self.cfg['backbone'])
        model, _, preprocess = open_clip.create_model_and_transforms(self.cfg['backbone'], pretrained='laion2b_s32b_b79k', device=self.device)
        #tokenizer = open_clip.get_tokenizer(self.cfg['backbone'])
        self.clip_model = model
        #self.tokenizer = tokenizer
        #self.clip_model.to(self.device).eval()

        # data
        self.train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.eval_tfm = preprocess
        self.train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=self.train_tfm, is_train=True, shuffle=False)
        self.val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=self.eval_tfm, shuffle=False)
        self.test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=self.eval_tfm, shuffle=False)
        self.metric = 'acc' if 'metric' not in self.cfg.keys() else self.cfg['metric']
        self.logger.info(f"Using metric {self.metric}...")

    def train_one_epoch(self, epoch):
        pass

    def train_mode(self):
        self.clip_model.tain()

    def eval_mode(self):
        self.clip_model.eval()
    
    def eval(self, split="test"):
        clip_weights = clip_classifier(self.dataset.classnames, self.dataset.template, self.clip_model, self.device)
        eval_loader = self.test_loader if split=='test' else self.val_loader
        test_features, test_labels = pre_load_features(self.cfg, 'test', self.clip_model, eval_loader, device=self.device)
        clip_logits = 100. * test_features @ clip_weights
        acc = cls_acc(clip_logits, test_labels, self.metric)
        if self.metric == 'f1' and split == 'test':
            c_acc = classwise_acc(clip_logits, test_labels)
            c_acc = [f"{a:.4f}" for a in c_acc]
            self.logger.info(f"Classwise acc: {','.join(c_acc)}")
        return acc
    
    def save(self, name):
        pass    # no model to save

    def load(self, name):
        pass    # no model to load