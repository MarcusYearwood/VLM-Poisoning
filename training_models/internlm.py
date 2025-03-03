import torch
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModel, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from accelerate.utils import gather_object
from parallelformers import parallelize
import warnings
from PIL import Image
import torchvision
from torchvision import transforms
from torch.nn import DataParallel
from torch.cuda.amp import autocast
import os
import numpy as np
import bitsandbytes as bnb

d_type=torch.float16

def get_image_encoder_internlm():
    vision_tower_name="internlm/internlm-xcomposer2d5-clip"
    warnings.filterwarnings("ignore")

    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name).eval().cuda()
    vision_tower.half()
    vision_tower.requires_grad_(True)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        vision_tower = DataParallel(vision_tower)

    img_size=560

    return vision_tower, None, None, img_size

def encode_image_internlm(image_encoder, X_adv, img_size, bs, diff_aug, orig_sizes):
    images = []
    for j in range(bs):
        orig_w, orig_h = orig_sizes[j]

        img = X_adv[j][:, :orig_h, :orig_w]
        img = __resize_img__(img)
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode='bicubic')

        if diff_aug:
            img = diff_aug(img).cuda()
        else:
            img = img.cuda()


        images.append(img)

    images = torch.cat(images, dim=0).cuda()

    with torch.autocast(device_type='cuda', dtype=d_type):
        image_embeds = image_encoder(images)

    return image_embeds.last_hidden_state

def __resize_img__(img):
    """Resize the image with padding to maintain aspect ratio."""
    _, h, w = img.shape
    target_size = max(h, w)
    
    # Calculate padding
    top_padding = (target_size - h) // 2
    bottom_padding = target_size - h - top_padding
    left_padding = (target_size - w) // 2
    right_padding = target_size - w - left_padding

    # Apply padding to make the image square
    padded_img = torchvision.transforms.functional.pad(
        img, [left_padding, top_padding, right_padding, bottom_padding]
    )

    return padded_img

def get_internlm_model():
    model_name="internlm/internlm-xcomposer2d5-7b"
    # model_name="internlm/internlm2-math-7b"
    d_type=torch.float16
    warnings.filterwarnings("ignore")
    # self.accelerator = Accelerator()

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Use 8-bit quantization
        llm_int8_threshold=200.0  # Adjust threshold for 8-bit quantization if necessary
    )

    device_map = { "": "cuda:1" } if torch.cuda.device_count() > 1 else 'auto'

    model = AutoModel.from_pretrained(
        model_name, 
        torch_dtype=d_type, 
        # quantization_config=quantization_config,
        trust_remote_code=True, 
        device_map=device_map
    )
    # self.model = self.accelerator.prepare(self.model)
    # parallelize(self.model, num_gpus=2, fp16=True, verbose='detail')
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.tokenizer = tokenizer
    num_beams = 3
    return model, tokenizer

def Image_transform(img, hd_num=25):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width/ height)
    scale = 1
    while scale*np.ceil(scale/ratio) <= hd_num:
        scale += 1
    scale -= 1
    scale = min(np.ceil(width / 560), scale)
    new_w = int(scale * 560)
    new_h = int(new_w / ratio)
    #print (scale, f'{height}/{new_h}, {width}/{new_w}')

    img = transforms.functional.resize(img, [new_h, new_w],)
    img = padding_336(img, 560)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img

def padding_336(b, pad=336):
    width, height = b.size
    tar = int(np.ceil(height / pad) * pad)
    top_padding = 0 # int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255,255,255])

    return b

def get_response_internlm(image, text_prompt, tokenizer, model):
        query = text_prompt
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                response, his = model.chat(tokenizer, query, transforms.ToTensor()(Image_transform(image)).unsqueeze(0).unsqueeze(0), do_sample=False, num_beams=3, use_meta=True)
        prediction = response.strip()
        return prediction