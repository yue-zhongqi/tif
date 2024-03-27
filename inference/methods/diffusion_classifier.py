import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
import math
from sklearn.metrics import f1_score
import tqdm
import numpy as np
import os
from datasets.utils import build_data_loader
from .base import BaseMethod
from .utils_diffusion import get_sd_model, get_scheduler_config
from .utils import classwise_accuracy_for_list


INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=interpolation),
        transforms.CenterCrop(size),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform

class DiffusionClassifier(BaseMethod):
    def __init__(self, config, dataset, device):
        super().__init__(config, dataset, device)
        interpolation = INTERPOLATIONS[config["interpolation"]]
        transform = get_transform(interpolation, config["img_size"])
        self.latent_size = config["img_size"] // 8

        # data
        self.train_loader = build_data_loader(data_source=dataset.train_x, batch_size=16, tfm=transform, is_train=True, shuffle=False)
        self.val_loader = build_data_loader(data_source=dataset.val, batch_size=1, is_train=False, tfm=transform, shuffle=True, return_id=True)
        self.test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=transform, shuffle=True, return_id=True)

        # load pretrained models
        self.logger.info(f"Loading SD version {config['version']}...")
        vae, tokenizer, text_encoder, unet, self.scheduler, self.pipeline = get_sd_model(config)
        self.vae = vae.to(device)
        self.text_encoder = text_encoder.to(device)
        self.unet = unet.to(device)
        self.tokenizer = tokenizer
        torch.backends.cudnn.benchmark = True
        
        # generate PDAE weight
        T = get_scheduler_config(self.cfg)['num_train_timesteps']
        self.T = T
        betas = torch.linspace(0.0001, 0.02, T)
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        if 'weight_t' in config.keys():
            if 'pdae' in config['weight_t']:
                pdae_gamma = float(config['weight_t'].split('-')[-1])
                self.logger.info(f"Using PDAE Weight with gamma {pdae_gamma}. Break down:")
                weights = snr ** pdae_gamma / (1 + snr)
                weights = weights / weights.max()   # so that max weight is 1
                self.logger.info(weights.numpy())
                weights = weights
            elif 'tif' in config['weight_t']:
                weights = self.get_tif_weights()
            else:
                weights = torch.ones(T)
        else:
            weights = torch.ones(T)
        self.weights = weights
        self.construct_text_embeddings()
        self.metric = 'acc' if 'metric' not in self.cfg.keys() else self.cfg['metric']
    
    def get_tif_weights(self):
        T = self.T
        betas = torch.linspace(0.0001, 0.02, T)
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        scale = alphas_cumprod.sqrt() / (2 * (2*((1-alphas_cumprod))).sqrt() )
        d = self.find_finegrain_edit()
        ts = range(0, T, 1)
        loss_fine = np.zeros(T, dtype=np.float64)
        loss_coarse = np.zeros(T, dtype=np.float64) + 1e-18
        ratio = np.zeros(T)
        for i in range(T):
            loss_fine[i] = (1-math.erf(scale[i].item() * d))
            for j in range(500):
                loss_coarse[i] += (1-math.erf(scale[i].item() * (d+1+j*1)))
            #loss_coarse[i] = (1-math.erf(t[i].item() * d_coarse)) + 1e-8
            ratio[i] = loss_fine[i] / loss_coarse[i]
        w = loss_fine / (loss_fine + loss_coarse)
        return torch.from_numpy(w.astype(np.float64))
    
    def find_finegrain_edit(self):
        train_loader = self.train_loader
        device = self.device
        embeds = []
        targets = []
        with torch.no_grad():
            for batch in train_loader:
                image, target = batch
                image = image.to(device).half()
                target = target.to(device)
                embeds.append(self.vae.encode(image).latent_dist.mean * 0.18215)
                targets.append(target)
        embeds = torch.cat(embeds, dim=0).float()
        targets = torch.cat(targets, dim=0)
        min_v = []
        same_class_map = (targets[:, None] == targets)
        for w in range(64):
            for h in range(64):
                distances = torch.cdist(embeds[:,:,w,h][None], embeds[:,:,w,h][None])
                distances = distances ** 2
                distances[0, same_class_map] = 100
                min_v.append(np.percentile(distances.cpu().numpy(), 0.1))
        return np.sqrt(np.array(min_v).sum())

    def construct_text_embeddings(self):
        tokenizer = self.tokenizer
        text_encoder = self.text_encoder
        # computing class prompt tokens & embeddings
        prompts = []
        prompt_labels = []
        prompt_class_names = []
        for i, classname in enumerate(self.dataset.classnames):
            texts = [t.format(self.get_classname(classname)) for t in self.dataset.template]
            if i == 0:
                self.logger.info(f"prompt examples of the first class: \"{','.join(texts)}\".")
            labels = [i for _ in self.dataset.template]
            names = [classname for _ in self.dataset.template]
            prompts += texts
            prompt_labels += labels
            prompt_class_names += names
        text_input = tokenizer(prompts, padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        embeddings = []
        with torch.inference_mode():
            for i in range(0, len(text_input.input_ids), 100):
                text_embeddings = text_encoder(
                    text_input.input_ids[i: i + 100].to(self.device),
                )[0]
                embeddings.append(text_embeddings)
        self.text_embeddings = torch.cat(embeddings, dim=0)
        self.prompt_labels = torch.from_numpy(np.array(prompt_labels)).to(self.device)
        self.prompt_class_names = prompt_class_names
        self.prompt_idx_to_classname = [self.prompt_class_names[i] for i in range(len(self.text_embeddings))]
        assert len(self.text_embeddings) == len(prompts)

    def get_classname(self, dataset_classname):
        return dataset_classname.replace('_', ' ')

    def train_one_epoch(self, epoch):
        pass

    def train_mode(self):
        pass

    def eval_mode(self):
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()
    
    def eval(self, split="test"):
        eval_loader = self.test_loader if split=='test' else self.val_loader
        correct = 0
        total = 0
        history = []
        tested = []
        wrong_stage = []
        preds = []
        gts = []
        if os.path.exists(f"{self.cfg['cache_dir']}/results.pt"):
            results = torch.load(f"{self.cfg['cache_dir']}/results.pt")
            correct = results['correct']
            total = results['total']
            history = results['history']
            tested = results['tested']
            wrong_stage = results['wrong_stage']
            if self.metric == 'f1':
                preds = results['preds']
                gts = results['gts']
            self.logger.info(f"Loaded {total} test results...")
        # pbar = tqdm(eval_loader)
        last_print = 0
        for i, (images, targets, idx) in enumerate(eval_loader):
            idx = idx.item()
            if idx in tested:
                self.logger.info(f"Skipping {idx}-th sample...")
                continue
            with torch.no_grad():
                image = images[0]
                target = targets[0]
                img_input = image.to(self.device).unsqueeze(0)
                if self.cfg["dtype"] == 'float16':
                    img_input = img_input.half()
                x0 = self.vae.encode(img_input).latent_dist.mean
                x0 *= 0.18215
            pred_idx, pred_errors, wrong = self.eval_prob_adaptive(
                self.unet, x0, self.text_embeddings, self.scheduler, self.cfg, self.latent_size, None, target
            )
            pred = self.prompt_labels[pred_idx]
            if pred == target:
                correct += 1
                history.append(1)
            else:
                history.append(0)
            if self.metric == 'f1':
                preds.append(pred.item())
                gts.append(target.item())
            total += 1
            tested.append(idx)
            wrong_stage.append(wrong)

            if self.metric == 'acc':
                acc = correct * 100 / float(total)
            else:
                acc = f1_score(np.array(gts), np.array(preds), average='macro') * 100.0
                c_acc = classwise_accuracy_for_list(preds, gts, len(self.dataset.classnames))
                c_acc = [f"{a:.4f}" for a in c_acc]
                self.logger.info(f"Classwise acc: {','.join(c_acc)}")

            self.logger.info(f"Img {i}/{len(eval_loader)}. Acc:{acc}.")
            torch.save({
                "correct": correct, "total": total, "history": history,
                "tested": tested, "wrong_stage": wrong_stage, "preds": preds, "gts": gts
            }, f"{self.cfg['cache_dir']}/results.pt")

            # last_print += 1
            # if last_print >= 10 and hasattr(self, 'misleading_t_tracker'):
            #     import matplotlib.pyplot as plt
            #     plt.figure(figsize=(16, 9), dpi=80)
            #     RESOLUTION = 1
            #     ts = range(0, self.T, RESOLUTION)

            #     ax1 = plt.subplot(2, 1, 1)
            #     ax1.set_title("misleading tracker")
            #     plt.bar(ts, self.misleading_t_tracker.numpy(), color='maroon', width=2.8)
            #     plt.xticks(ticks=range(0,self.T,50), labels=range(0,self.T,50))

            #     ax2 = plt.subplot(2, 1, 2)
            #     ax2.set_title("discriminative tracker")
            #     plt.bar(ts, self.discriminative_t_tracker.numpy(), color='maroon', width=2.8)
            #     plt.xticks(ticks=range(0,self.T,50), labels=range(0,self.T,50))

            #     plt.savefig(os.path.join(self.cfg['cache_dir'], "probe.png"))
            #     last_print = 0
        if self.metric == 'acc':
            acc = correct * 100 / float(total)
        else:
            acc = f1_score(np.array(gts), np.array(preds), average='macro') * 100.0

        return acc
    
    def save(self, name):
        pass    # no model to save

    def load(self, name):
        pass    # no model to load


    def eval_prob_adaptive(self, unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None, target=-1):
        scheduler_config = get_scheduler_config(args)
        T = scheduler_config['num_train_timesteps']
        max_n_samples = max(args["n_samples"])

        if all_noise is None:
            all_noise = torch.randn((max_n_samples * args["n_trials"], 4, latent_size, latent_size), device=latent.device)
        if args["dtype"] == 'float16':
            all_noise = all_noise.half()
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

        data = dict()
        t_evaluated = set()
        remaining_prmpt_idxs = list(range(len(text_embeds)))
        start = T // max_n_samples // 2
        t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

        for n_samples, n_to_keep in zip(args["n_samples"], args["to_keep"]):
            ts = []
            noise_idxs = []
            text_embed_idxs = []
            curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
            curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
            for prompt_i in remaining_prmpt_idxs:
                for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                    ts.extend([t] * args["n_trials"])
                    noise_idxs.extend(list(range(args["n_trials"] * t_idx, args["n_trials"] * (t_idx + 1))))
                    text_embed_idxs.extend([prompt_i] * args["n_trials"])
            t_evaluated.update(curr_t_to_eval)
            pred_errors = self.eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                    text_embeds, text_embed_idxs, args["batch_size"], args["dtype"], args["loss"], latent.device)
            # match up computed errors to the data
            for prompt_i in remaining_prmpt_idxs:
                mask = torch.tensor(text_embed_idxs) == prompt_i
                prompt_ts = torch.tensor(ts)[mask]
                prompt_pred_errors = pred_errors[mask]
                if prompt_i not in data:
                    data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
                else:
                    data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                    data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

            # compute the next remaining idxs
            errors = [-(data[prompt_i]['pred_errors'] * self.weights[data[prompt_i]['t']]).mean() for prompt_i in remaining_prmpt_idxs]
            best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
            remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

        # organize the output
        assert len(remaining_prmpt_idxs) == 1
        pred_idx = remaining_prmpt_idxs[0]
        return pred_idx, data, -1


    def eval_error(self, unet, scheduler, latent, all_noise, ts, noise_idxs,
                text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2', device='cpu'):
        assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
        pred_errors = torch.zeros(len(ts), device='cpu')
        idx = 0
        with torch.inference_mode():
            for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
                batch_ts = torch.tensor(ts[idx: idx + batch_size])
                noise = all_noise[noise_idxs[idx: idx + batch_size]]
                noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                                noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
                t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
                text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
                noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
                if loss == 'l2':
                    error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                elif loss == 'l1':
                    error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                elif loss == 'huber':
                    error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                else:
                    raise NotImplementedError
                pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
                idx += len(batch_ts)
        return pred_errors