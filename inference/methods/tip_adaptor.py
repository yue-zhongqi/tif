import clip
import torchvision.transforms as transforms
from methods.utils import clip_classifier, pre_load_features, cls_acc, search_hp, classwise_acc
from datasets.utils import build_data_loader
from .base import BaseMethod
import open_clip
from .zs_clip import ZeroShotCLIP
from .utils import build_cache_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class TipAdaptor(ZeroShotCLIP):
    def __init__(self, config, dataset, device):
        super().__init__(config, dataset, device)
        self.logger.info(f"Running TipAdaptor at {self.cfg['shots']} shots...")
        # data
        self.train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=self.train_tfm, is_train=True, shuffle=True)
        # pre load features and weights
        self.clip_weights = clip_classifier(self.dataset.classnames, self.dataset.template, self.clip_model, self.device)
        self.cache_keys, self.cache_values = build_cache_model(config, self.clip_model, self.train_loader, self.device)
        self.val_features, self.val_labels = pre_load_features(config, "val", self.clip_model, self.val_loader, self.device)
        self.test_features, self.test_labels = pre_load_features(config, "test", self.clip_model, self.test_loader, self.device) 
        # to handle custom classes
        if 'clip_weight' in self.cfg.keys():
            self.clip_weight = self.cfg["clip_weight"]
        else:
            self.clip_weight = 1.0
        self.metric = 'acc' if 'metric' not in self.cfg.keys() else self.cfg['metric']
        self.logger.info(f"Using metric {self.metric}...")

    def train_one_epoch(self, epoch):
        pass

    def train_mode(self):
        self.clip_model.train()

    def eval_mode(self):
        self.clip_model.eval()
    
    def eval(self, split="test"):
        best_beta, best_alpha = search_hp(
            self.cfg, self.cache_keys, self.cache_values,
            self.val_features, self.val_labels, self.clip_weights, metric=self.metric,
        )
        clip_logits = 100. * self.test_features @ self.clip_weights
        # Tip-Adapter    
        affinity = self.test_features @ self.cache_keys
        cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ self.cache_values
        
        tip_logits = self.clip_weight * clip_logits + cache_logits * best_alpha
        acc = cls_acc(tip_logits, self.test_labels, self.metric)
        if self.metric == 'f1' and split == 'test':
            c_acc = classwise_acc(tip_logits, self.test_labels)
            c_acc = [f"{a:.4f}" for a in c_acc]
            self.logger.info(f"Classwise acc: {','.join(c_acc)}")
        return acc
    
    def save(self, name):
        pass    # no model to save

    def load(self, name):
        pass    # no model to load


class TipAdaptorF(TipAdaptor):
    def __init__(self, config, dataset, device):
        super().__init__(config, dataset, device)
        self.adapter = nn.Linear(
            self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False
        ).to(self.device)
        self.adapter.weight = nn.Parameter(self.cache_keys.t())
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.cfg['lr'], eps=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.cfg['train_epochs'] * len(self.train_loader_F)
        )
        self.beta, self.alpha = self.cfg['init_beta'], self.cfg['init_alpha']

    def train_mode(self):
        self.clip_model.train()
        self.adapter.train()

    def eval_mode(self):
        self.clip_model.eval()
        self.adapter.eval()

    def train_one_epoch(self, epoch):
        correct_samples, all_samples = 0, 0
        loss_list = []
        self.logger.info('Train Epoch: {:} / {:}'.format(epoch, self.cfg['train_epochs']))

        for i, (images, target) in enumerate(tqdm(self.train_loader_F)):
            images, target = images.to(self.device), target.to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = self.adapter(image_features)
            cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values
            clip_logits = 100. * image_features @ self.clip_weights
            tip_logits = self.clip_weight * clip_logits + cache_logits * self.alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target, self.metric)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def eval(self, split='test'):
        eval_features = self.test_features if split=='test' else self.val_features
        eval_labels = self.test_labels if split=='test' else self.val_labels
        if split == 'test':
            beta, alpha = search_hp(
                self.cfg, self.cache_keys, self.cache_values,
                self.val_features, self.val_labels, self.clip_weights, metric=self.metric,
                adapter=self.adapter
            )
        else:
            beta, alpha = self.beta, self.alpha

        affinity = self.adapter(eval_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
        clip_logits = 100. * eval_features @ self.clip_weights
        tip_logits = self.clip_weight * clip_logits + cache_logits * alpha
        if self.metric == 'f1' and split == 'test':
            c_acc = classwise_acc(tip_logits, eval_labels)
            c_acc = [f"{a:.4f}" for a in c_acc]
            self.logger.info(f"Classwise acc: {','.join(c_acc)}")
        return cls_acc(tip_logits, eval_labels, self.metric)
    
    def save(self, name):
        torch.save(self.adapter.weight, f"{self.cfg['cache_dir']}/{name}.pt")

    def load(self, name):
        self.adapter.weight = torch.load(f"{self.cfg['cache_dir']}/{name}.pt")