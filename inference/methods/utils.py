from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import clip
import json
from open_clip import tokenizer
from sklearn.metrics import f1_score


def cls_acc(output, target, metric, topk=1):
    if metric == 'acc':
        pred = output.topk(topk, 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        acc = 100 * acc / target.shape[0]
    elif metric == 'f1':
        acc = 0
        _, pred = torch.max(output, dim=1)
        predictions_np = pred.detach().cpu().numpy()
        ground_truth_np = target.detach().cpu().numpy()
        acc = f1_score(ground_truth_np, predictions_np, average='macro')
    return acc

def classwise_acc(output, ground_truth):
    _, predictions = torch.max(output, dim=1)
    num_classes = len(torch.unique(ground_truth))
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    acc = []
    for i in range(num_classes):
        correct[i] = torch.sum((predictions == i) & (ground_truth == i))
        total[i] = torch.sum(ground_truth == i)
        if total[i] == 0:
            acc.append(-1)
        else:
            acc.append((correct[i] / total[i]).item())
    return acc

def classwise_accuracy_for_list(predictions, ground_truth, num_classes):
    correct = [0] * num_classes
    total = [0] * num_classes

    # Iterate over the predictions and ground truth
    for pred, truth in zip(predictions, ground_truth):
        total[truth] += 1
        if pred == truth:
            correct[truth] += 1

    # Calculate accuracy for each class
    accuracy = []
    for i in range(num_classes):
        if total[i] == 0:
            acc = -1
        else:
            acc = correct[i] / total[i]
        accuracy.append(acc)
    return accuracy

def clip_classifier(classnames, template, clip_model, device=None):
    if device is None:
        device = "cuda:0"

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = tokenizer.tokenize(texts).to(device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts).float()
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache, device):
    device = "cuda:0" if device is None else device
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.to(device)
                    image_features = clip_model.encode_image(images).float()
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.to(device)
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).float()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt").to(device)
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt").to(device)

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader, device=None):
    device = "cuda:0" if device is None else device
    f_name = cfg['cache_dir'] + "/" + split + "_f.pt"
    l_name = cfg['cache_dir'] + "/" + split + "_l.pt"
    if os.path.exists(f_name) and os.path.exists(l_name):
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt", map_location=device)
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt", map_location=device)
    else:
        features, labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.to(device), target.to(device)
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, metric, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels, metric)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


### dictionary to object
class Obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=Obj)