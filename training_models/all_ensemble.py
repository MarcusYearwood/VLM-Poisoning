from transformers import AutoModel
from torch import nn

import torchvision
from PIL import Image
import torch
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(".."))  # Adjust as needed

from models.CLIP.clip import clip

import warnings

class MyEnsemble():
    def __init__(self, tau=2):
        warnings.filterwarnings("ignore")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_rn_50,_ = clip.load('RN50', device=self.device)
        clip_rn_101,_ = clip.load('RN101', device=self.device)
        clip_vit_b_16,_ = clip.load('ViT-B/16', device=self.device)
        clip_vit_b_32,_ = clip.load('ViT-B/32', device=self.device)
        clip_vit_l_14,_ = clip.load('ViT-L/14', device=self.device)
        self.clip_models = [clip_rn_50, clip_rn_101, clip_vit_b_16, clip_vit_b_32, clip_vit_l_14, ]
        self.intern_models = [
            AutoModel.from_pretrained(
            'OpenGVLab/InternViT-300M-448px-V2_5',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).to(self.device).eval(),

            AutoModel.from_pretrained(
            'OpenGVLab/InternViT-300M-448px',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).to(self.device).eval(),

            # AutoModel.from_pretrained(
            # 'OpenGVLab/InternViT-6B-448px-V1-5',
            # torch_dtype=torch.bfloat16,
            # low_cpu_mem_usage=True,
            # trust_remote_code=True).to(self.device).eval()
        ]
        self.models_num = len(self.clip_models) + len(self.intern_models)

        self.intern_preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(448, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
                torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
                torchvision.transforms.CenterCrop(448),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
            ]
        )

        self.clip_preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(clip_vit_b_32.visual.input_resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
                torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
                torchvision.transforms.CenterCrop(clip_vit_b_32.visual.input_resolution),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
            ]
        )

        self.costs = torch.ones(2, self.models_num).to(self.device)
        self.tau = tau
        self.critical = nn.CosineSimilarity(dim=0, eps=1e-6).to(self.device)


        
    def encode_image(self, image, use_grad=True):
        image_features_list = []
        # image_tgt = self.preprocess(image).to(torch.float32)

        context = torch.enable_grad() if use_grad else torch.no_grad()
        with context:
            for clip_model in self.clip_models:
                image_features = clip_model.encode_image(self.clip_preprocess(image))  # [bs, 512]
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                image_features_list.append(image_features)
            for model in self.intern_models:
                image_features = model(self.intern_preprocess(image).to(torch.bfloat16)).last_hidden_state  # [bs, 512]
                image_features = image_features / image_features.norm()
                image_features_list.append(image_features)

        return image_features_list
    
    def get_gradients(self, adv_image_features_list, tgt_image_features_list, adv_tensor):
        model_losses=torch.zeros(self.models_num)
        loss = torch.zeros(1).to(self.device)
        model_losses = torch.stack([torch.norm(adv_embed - tgt_embed) for adv_embed, tgt_embed in zip(adv_image_features_list, tgt_image_features_list)])

        exp_cost_ratio = torch.exp(self.tau*(self.costs[1] / self.costs[0]+1e-16))
        weights = torch.sum(exp_cost_ratio, dim=0) / (self.models_num*exp_cost_ratio)
        loss = torch.sum(weights * model_losses)

        self.costs[1] = self.costs[0]
        self.costs[0] = model_losses.clone().detach()


        gradient = -torch.autograd.grad(loss, adv_tensor)[0]
        return gradient, torch.mean(model_losses), model_losses