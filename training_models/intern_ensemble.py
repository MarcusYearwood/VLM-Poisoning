import torch
from PIL import Image
from transformers import AutoModel
from torch import nn
import torchvision

import warnings

class MyInternEnsemble():
    def __init__(self, tau=2):
        warnings.filterwarnings("ignore")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [
            AutoModel.from_pretrained(
            'OpenGVLab/InternViT-300M-448px-V2_5',
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True).to(self.device).eval(),

            # AutoModel.from_pretrained(
            # 'OpenGVLab/InternViT-6B-448px-V2_5',
            # torch_dtype=torch.bfloat16,
            # low_cpu_mem_usage=True,
            # trust_remote_code=True).to(self.device).eval(),

            # AutoModel.from_pretrained(
            # 'OpenGVLab/InternViT-6B-448px-V1-5',
            # torch_dtype=torch.bfloat16,
            # low_cpu_mem_usage=True,
            # trust_remote_code=True).to(self.device).eval()
        ]

        self.preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(448, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
                torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
                torchvision.transforms.CenterCrop(448),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
            ]
        )

        self.costs = torch.ones(2, len(self.models)).to(self.device)
        self.tau = tau
        self.critical = nn.CosineSimilarity(dim=0, eps=1e-6).to(self.device)


        
    def encode_image(self, image, use_grad=True):
        image_features_list = []
        image_tgt = self.preprocess(image).to(torch.float32)

        context = torch.enable_grad() if use_grad else torch.no_grad()
        with context:
            for model in self.models:
                image_features = model(image_tgt).last_hidden_state  # [bs, 512]
                image_features = image_features / image_features.norm()
                image_features_list.append(image_features)

        return image_features_list
    
    def get_gradients(self, adv_image_features_list, tgt_image_features_list, adv_tensor):
        model_losses=torch.zeros(len(self.models))
        loss = torch.zeros(1).to(self.device)
        model_losses = torch.stack([self.critical(adv_embed.view(-1), tgt_embed.view(-1)) for adv_embed, tgt_embed in zip(adv_image_features_list, tgt_image_features_list)])

        exp_cost_ratio = torch.exp(self.tau*(self.costs[1] / self.costs[0]+1e-16))
        weights = torch.sum(exp_cost_ratio, dim=0) / (len(self.models)*exp_cost_ratio)
        loss = torch.sum(weights * model_losses)

        self.costs[1] = self.costs[0]
        self.costs[0] = model_losses.clone().detach()


        gradient = torch.autograd.grad(loss, adv_tensor)[0]
        return gradient, torch.mean(model_losses)