"""
import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPVisionModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoTokenizer
from PIL import Image
import torchvision
from torch.nn import DataParallel
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import warnings

d_type = torch.float16

clip_preprocess = torchvision.transforms.Compose( # from attack vlm
    [
        T.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
        T.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
        T.CenterCrop(224),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
    ]
)

def get_image_encoder_clip():
    vision_tower_name = "openai/clip-vit-large-patch14"
    warnings.filterwarnings("ignore")

    vision_tower, preprocess = clip.load('ViT-B/32').eval().cuda()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        vision_tower = vision_tower

    img_size = 224  # Standard image size for CLIP

    processor = CLIPTokenizer.from_pretrained(vision_tower_name)

    return vision_tower, processor, None, img_size

def encode_image_clip(image_encoder, X_adv, img_size, bs, diff_aug, orig_sizes):
    images = X_adv

    with torch.autocast(device_type='cuda', dtype=d_type):
        image_embeds = image_encoder.encode_image(images)
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

    return image_embeds


def i2t_similarity_clip(image_encoder, processor, image_tensor, text):
    # Preprocess image and text

    image = clip_preprocess(image_tensor.squeeze(0)).unsqueeze(0)

    text_inputs = processor([text], return_tensors='pt', padding=True, truncation=True)

    # inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    # Get image and text embeddings
    # inputs = {key: value.cuda() for key, value in inputs.items()}
    outputs = image_encoder(input_ids=text_inputs['input_ids'].cuda(),
                attention_mask=text_inputs['attention_mask'].cuda(),
                pixel_values=image.cuda())

    similarity = outputs.logits_per_image
    return similarity

class MyClip():
    def __init__(self):
        warnings.filterwarnings("ignore")
        model_name = "openai/clip-vit-base-patch32"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(self.device)
        self.clip_text_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.input_res = self.clip_vision_model.vision_model.config.image_size

    def encode_image(self, image):
        outputs = self.clip_vision_model(pixel_values=clip_preprocess(image))
        return outputs.image_embeds

    def encode_text(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        inputs.to(self.device)
        outputs = self.clip_text_model(**inputs)
        return outputs.text_embeds
"""

import torchvision
from PIL import Image
import torch
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(".."))  # Adjust as needed

from models.CLIP.clip import clip

import warnings

# models = [clip_vit_b_32]

# final_preprocess = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.Resize(clip_vit_b_32.visual.input_resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
#         # torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
#         torchvision.transforms.CenterCrop(clip_vit_b_32.visual.input_resolution),
#     ]
# )

class MyClipEnsemble():
    def __init__(self, tau=2):
        warnings.filterwarnings("ignore")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_rn_50,_ = clip.load('RN50', device=self.device)
        clip_rn_101,_ = clip.load('RN101', device=self.device)
        clip_vit_b_16,_ = clip.load('ViT-B/16', device=self.device)
        clip_vit_b_32,_ = clip.load('ViT-B/32', device=self.device)
        clip_vit_l_14,_ = clip.load('ViT-L/14', device=self.device)
        self.models = [clip_rn_50, clip_rn_101, clip_vit_b_16, clip_vit_b_32, clip_vit_l_14]

        self.clip_preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(clip_vit_b_32.visual.input_resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
                # torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
                torchvision.transforms.CenterCrop(clip_vit_b_32.visual.input_resolution),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
            ]
        )

        self.costs = torch.ones(2, len(self.models))
        self.tau = tau
        
    def encode_image(self, image, use_grad=True):
        image_features_list = []
        image_tgt = self.clip_preprocess(image)

        context = torch.enable_grad() if use_grad else torch.no_grad()
        with context:
            for clip_model in self.models:
                image_features = clip_model.encode_image(image_tgt)  # [bs, 512]
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                image_features_list.append(image_features)

        return image_features_list
    
    def get_gradients(self, adv_image_features_list, tgt_image_features_list, adv_tensor):
        model_losses=torch.zeros(len(self.models))
        loss = torch.zeros(1).to(self.device)
        for model_i, (pred_i, target_i) in enumerate(zip(adv_image_features_list, tgt_image_features_list)):
            model_losses[model_i] = torch.mean(torch.sum(pred_i * target_i, dim=1))   

        exp_cost_ratio = torch.exp(self.tau*(self.costs[1] / self.costs[0]+1e-16))
        weights = torch.sum(exp_cost_ratio, dim=0) / (len(self.models)*exp_cost_ratio)
        loss = torch.sum(weights * model_losses)

        self.costs[1] = self.costs[0]
        self.costs[0] = model_losses.clone().detach()


        gradient = torch.autograd.grad(loss, adv_tensor)[0]
        return gradient, torch.mean(model_losses)
