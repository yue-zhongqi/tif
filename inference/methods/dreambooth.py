import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F

import os
from tqdm import tqdm, trange
import numpy as np
from pathlib import Path
import hashlib
import itertools
import math

from datasets.utils import build_data_loader
from .diffusion_classifier import DiffusionClassifier
from .utils_diffusion import get_sd_model, get_scheduler_config, MODEL_IDS
from .utils import dict2obj
from .utils_dreambooth_lora import PromptDataset, DreamBoothDataset, get_display_images
from .lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    safetensors_available,
    save_lora_weight,
    save_safeloras,
    tune_lora_scale,
    monkeypatch_or_replace_lora,
    load_all_loras

)
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
import re


class Dreambooth(DiffusionClassifier):
    def __init__(self, config, dataset, device):
        args = dict2obj(config)
        self.args = args
        super().__init__(config, dataset, device)
        self.logger.info(f"Using cutoff T: {self.args.cutoff_T}.")

        # init unet filtering fn
        if args.unet_lora_filter == 'none':
            filter_fn = None
        else:
            self.logger.info(f"Applying UNET lora filter {args.unet_lora_filter}...")
            mode, lvl = args.unet_lora_filter.split('-')[0], int(args.unet_lora_filter.split('-')[1])
            down_conds = {0: 'down_blocks.(0|1|2)',1:'down_blocks.(1|2)',2:'down_blocks.2'}
            up_conds = {0: 'up_blocks.(1|2|3)',1:'up_blocks.(1|2)',2:'up_blocks.1'}
            mid_cond = 'mid_block'
            conds = {'down':[down_conds[lvl]],'mirror':[down_conds[lvl],up_conds[lvl]],'all':[down_conds[lvl],up_conds[lvl],mid_cond]}
            def filter_fn(name):
                return any(re.search(regex, name) for regex in conds[mode])
        # load all unet loras in one go
        self.loras_list = load_all_loras(
            self.unet, self.args.lora_dir, self.args.iteration,
            r=self.args.lora_rank, scale=self.args.unet_lora_scale,
            filtering_fn=filter_fn
        )
        scheduler_config = get_scheduler_config(self.cfg)
        self.T = scheduler_config['num_train_timesteps']
        # tracking
        self.misleading_t_tracker = torch.zeros(self.T).cpu()
        self.discriminative_t_tracker = torch.zeros(self.T).cpu()

        # eval method
        if 'test_method' in self.cfg.keys() and self.cfg['test_method'] == 'sampling':
            self.logger.info("Test by sampling...")
            self.eval_prob_adaptive = self.eval_prob_adaptive_tif
        else:
            self.logger.info("Test by re-weighting...")
            self.eval_prob_adaptive = self.eval_prob_adaptive_diffclf


    def get_lora_dirs(self, classname):
        classname_dir = classname.replace("/", "-")
        classname_dir = '_'.join(classname_dir.split(' '))
        return f"{self.args.lora_dir}/{classname_dir}/unet_{self.args.iteration}.pt", \
            f"{self.args.lora_dir}/{classname_dir}/text_{self.args.iteration}.pt"
                
    def construct_text_embeddings(self):
        tokenizer = self.tokenizer
        text_encoder = self.text_encoder
        # computing class prompt tokens & embeddings
        prompts = []
        prompt_labels = []
        prompt_class_names = []
        embeddings = []
        for i, dset_classname in enumerate(self.dataset.classnames):
            classname = dset_classname.replace('_', ' ')
            _, text_lora_dir = self.get_lora_dirs(classname)
            # texts = [t.format(classname) for t in self.dataset.template]
            if self.args.add_class_name:
                texts = [t.format(self.args.lora_token + " " + dset_classname) \
                         for t in self.dataset.template] if self.args.lora_token != 'none' else \
                        [t.format(dset_classname) for t in self.dataset.template]
            else:
                texts = [t.format(self.args.lora_token) for t in self.dataset.template]
            if i == 0:
                self.logger.info(f"prompt examples of the first class: \"{','.join(texts)}\".")
            text_input = tokenizer(texts, padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            monkeypatch_or_replace_lora(text_encoder, torch.load(text_lora_dir), target_replace_module=["CLIPAttention"], r=self.args.lora_rank)
            tune_lora_scale(text_encoder, self.args.text_lora_scale)
            with torch.inference_mode():
                text_embeddings = text_encoder(
                    text_input.input_ids.to(self.device),
                )[0]
            embeddings.append(text_embeddings)
            labels = [i for _ in self.dataset.template]
            names = [classname for _ in self.dataset.template]
            prompts += texts
            prompt_labels += labels
            prompt_class_names += names
        self.text_embeddings = torch.cat(embeddings, dim=0)
        self.prompt_labels = torch.from_numpy(np.array(prompt_labels)).to(self.device)
        self.prompt_class_names = prompt_class_names
        self.prompt_idx_to_classname = [self.prompt_class_names[i] for i in range(len(self.text_embeddings))]
        assert len(self.text_embeddings) == len(prompts)

    def eval_prob_adaptive_tif(self, unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None, target=-1):
        T = self.T
        max_n_samples = sum(args["n_samples"])

        if all_noise is None:
            all_noise = torch.randn((max_n_samples * args["n_trials"], 4, latent_size, latent_size), device=latent.device)
        if args["dtype"] == 'float16':
            all_noise = all_noise.half()
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

        data = dict()
        t_evaluated = 0
        remaining_prmpt_idxs = list(range(len(text_embeds)))
        
        ts = np.arange(T)
        w = self.weights.numpy()
        t_to_eval = np.random.choice(ts, p=w/w.sum(), size=max_n_samples)
        # t_to_eval = list(range(10, 200, 10))
        stage = -1
        wrong_stage = -1
        for n_samples, n_to_keep in zip(args["n_samples"], args["to_keep"]):
            stage += 1
            ts = []
            noise_idxs = []
            text_embed_idxs = []
            curr_t_to_eval = t_to_eval[t_evaluated:t_evaluated+n_samples]
            # curr_t_to_eval = t_to_eval
            # print(len(curr_t_to_eval))
            for prompt_i in remaining_prmpt_idxs:
                for t_idx, t in enumerate(curr_t_to_eval, start=t_evaluated):
                    ts.extend([t] * args["n_trials"])
                    noise_idxs.extend(list(range(args["n_trials"] * t_idx, args["n_trials"] * (t_idx + 1))))
                    text_embed_idxs.extend([prompt_i] * args["n_trials"])
            t_evaluated += n_samples
            pred_errors = self.eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                    text_embeds, text_embed_idxs, len(curr_t_to_eval), args["batch_size"], args["dtype"], args["loss"], latent.device)
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
            errors = [-(data[prompt_i]['pred_errors']).mean() for prompt_i in remaining_prmpt_idxs]
            best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
            last_remaining_prmpt_idxs = remaining_prmpt_idxs
            remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]
            remaining_preds = [self.prompt_labels[idx] for idx in remaining_prmpt_idxs]
            if target not in remaining_preds and wrong_stage < 0:
                wrong_stage = stage
            # update tracking info
            if wrong_stage > 0:
                # already wrong
                target_prompt_idx = [idx for idx in range(len(text_embeds)) if self.prompt_labels[idx] == target][0]
                self.misleading_t_tracker[data[target_prompt_idx]['t']] += \
                    data[target_prompt_idx]['pred_errors'] - data[last_remaining_prmpt_idxs[best_idxs[0]]]['pred_errors']
                return best_idxs[0], data, wrong_stage
        # still correct
        errors = [-(data[prompt_i]['pred_errors']).mean() for prompt_i in last_remaining_prmpt_idxs]
        top2_indices = torch.topk(torch.tensor(errors), k=2, dim=0).indices.tolist()
        target_prompt_idx = last_remaining_prmpt_idxs[top2_indices[0]]
        most_misleading_idx = last_remaining_prmpt_idxs[top2_indices[1]]
        self.discriminative_t_tracker[data[target_prompt_idx]['t']] += \
            data[most_misleading_idx]['pred_errors'] - data[target_prompt_idx]['pred_errors']
        
        # organize the output
        assert len(remaining_prmpt_idxs) == 1
        pred_idx = remaining_prmpt_idxs[0]
        return pred_idx, data, wrong_stage
    
    def eval_prob_adaptive_diffclf(self, unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None, target=-1):
        T = self.T
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
        # t_to_eval = list(range(10, 200, 10))
        stage = -1
        wrong_stage = -1
        for n_samples, n_to_keep in zip(args["n_samples"], args["to_keep"]):
            stage += 1
            ts = []
            noise_idxs = []
            text_embed_idxs = []
            curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
            curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
            # curr_t_to_eval = t_to_eval
            # print(len(curr_t_to_eval))
            for prompt_i in remaining_prmpt_idxs:
                for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                    ts.extend([t] * args["n_trials"])
                    noise_idxs.extend(list(range(args["n_trials"] * t_idx, args["n_trials"] * (t_idx + 1))))
                    text_embed_idxs.extend([prompt_i] * args["n_trials"])
            t_evaluated.update(curr_t_to_eval)
            pred_errors = self.eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                    text_embeds, text_embed_idxs, len(curr_t_to_eval), args["batch_size"], args["dtype"], args["loss"], latent.device)
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
            last_remaining_prmpt_idxs = remaining_prmpt_idxs
            remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]
            remaining_preds = [self.prompt_labels[idx] for idx in remaining_prmpt_idxs]
            if target not in remaining_preds and wrong_stage < 0:
                wrong_stage = stage
            # update tracking info
            if wrong_stage > 0:
                # already wrong
                target_prompt_idx = [idx for idx in range(len(text_embeds)) if self.prompt_labels[idx] == target][0]
                self.misleading_t_tracker[data[target_prompt_idx]['t']] += \
                    data[target_prompt_idx]['pred_errors'] - data[last_remaining_prmpt_idxs[best_idxs[0]]]['pred_errors']
                return best_idxs[0], data, wrong_stage
        # still correct
        errors = [-(data[prompt_i]['pred_errors'] * self.weights[data[prompt_i]['t']]).mean() for prompt_i in last_remaining_prmpt_idxs]
        top2_indices = torch.topk(torch.tensor(errors), k=2, dim=0).indices.tolist()
        target_prompt_idx = last_remaining_prmpt_idxs[top2_indices[0]]
        most_misleading_idx = last_remaining_prmpt_idxs[top2_indices[1]]
        self.discriminative_t_tracker[data[target_prompt_idx]['t']] += \
            data[most_misleading_idx]['pred_errors'] - data[target_prompt_idx]['pred_errors']
        
        # organize the output
        assert len(remaining_prmpt_idxs) == 1
        pred_idx = remaining_prmpt_idxs[0]
        return pred_idx, data, wrong_stage
    
    # ts, all_noise, text_embeds_idx
    def eval_error(self, unet, scheduler, latent, all_noise, ts, noise_idxs,
                text_embeds, text_embed_idxs, group_size, batch_size=32, dtype='float32', loss='l2', device='cpu'):
        assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
        pred_errors = torch.zeros(len(ts), device='cpu')
        ts, noise_idxs, text_embed_idxs = torch.tensor(ts), torch.tensor(noise_idxs), torch.tensor(text_embed_idxs)

        # cutoff large timesteps
        if self.args.cutoff_T > 0:
            ts[ts > self.args.cutoff_T] = 350
            assert (ts<0).sum() == 0

        ts_classwise, noise_idxs_classwise, text_embeds_classwise = ts.view(-1, group_size, *ts.shape[2:]), \
            noise_idxs.view(-1, group_size, *noise_idxs.shape[2:]), text_embed_idxs.view(-1, group_size, *text_embed_idxs.shape[2:])
        for i in range(ts_classwise.shape[0]):
            idx = 0
            ts, noise_idxs, text_embed_idxs = ts_classwise[i], noise_idxs_classwise[i], text_embeds_classwise[i]
            assert text_embed_idxs[0] == text_embed_idxs[-1]
            # unet_lora_dir, _ = self.get_lora_dirs(self.prompt_idx_to_classname[text_embed_idxs[0]])
            cname = self.prompt_idx_to_classname[text_embed_idxs[0]]
            cname = cname.replace("/", "-")
            cname = '_'.join(cname.split(' '))
            for lora in self.loras_list:
                lora.name = cname
            #monkeypatch_or_replace_lora(unet, torch.load(unet_lora_dir))
            #tune_lora_scale(unet, self.args.unet_lora_scale)
            with torch.inference_mode():
                for _ in trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
                    end_idx = min(idx + batch_size, len(ts))
                    batch_ts = ts[idx: end_idx]
                    noise = all_noise[noise_idxs[idx: end_idx]]
                    noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                                    noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
                    t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
                    text_input = text_embeds[text_embed_idxs[idx: end_idx]]
                    noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
                    if loss == 'l2':
                        error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                    elif loss == 'l1':
                        error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                    elif loss == 'huber':
                        error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                    else:
                        raise NotImplementedError
                    pred_errors[idx + i * len(ts): end_idx + i * len(ts)] = error.detach().cpu()
                    idx = end_idx
        return pred_errors
        