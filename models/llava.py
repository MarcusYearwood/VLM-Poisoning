import gc
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import copy
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# llava
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

"""
Given a list of image tensors, a model's image encoder should resize and encode the images 
"""
def get_image_encoder_llava():
      '''
      Return: the image encoder, image processor and the data augmention used during training

      image_processor is only for sanity check in test_attack_efficacy()
      diff_aug will be used in crafting adversarial examples
      '''
      model_path = "liuhaotian/llava-v1.5-7b"
      tokenizer, model, image_processor, context_len = load_pretrained_model(
      model_path=model_path,
      model_base=None,
      model_name=get_model_name_from_path(model_path)
      )

      vision_model = copy.deepcopy(model.model.vision_tower); vision_model.eval()
      # In llava, the forward function of CLIP is wrapped with torch.no_grad, which we get rid of below
      image_encoder_ = vision_model.forward.__wrapped__
      image_encoder = lambda x: image_encoder_(vision_model, x)

      # delete the model (including LLM) to save memory
      del model
      gc.collect(); torch.cuda.empty_cache()

      diff_aug = None

      img_size = 336

      return image_encoder, image_processor, diff_aug, img_size

def encode_image_llava(image_encoder, X_adv, img_size, bs, diff_aug, orig_sizes, normalize=normalize):
    X_adv_resized = torch.cat([
    TF.resize(
            X_adv[j][:, :orig_sizes[j][1], :orig_sizes[j][0]], 
            (img_size, img_size), 
            interpolation=InterpolationMode.BICUBIC
            ).unsqueeze(0) for j in range(bs)
    ], dim=0)

    if diff_aug is not None:
            # NOTE: using differentiable randomresizedcrop here 
            X_adv_input_to_model = normalize(diff_aug(X_adv_resized))
    else:
            X_adv_input_to_model = normalize(X_adv_resized)

    return image_encoder(X_adv_input_to_model)