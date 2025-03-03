import gc
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import copy
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.nn import DataParallel
import inspect

from transformers import CLIPImageProcessor

# llava
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria, get_model_name_from_path

import contextlib

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

      def forward_without_no_grad(images):
            if type(images) is list:
                  image_features = []
                  for image in images:
                        image_forward_out = vision_model.vision_tower(
                              image.to(device=vision_model.device, dtype=vision_model.dtype).unsqueeze(0),
                              output_hidden_states=True
                        )
                        image_feature = vision_model.feature_select(image_forward_out).to(image.dtype)
                        image_features.append(image_feature)
            else:
                  image_forward_outs = vision_model.vision_tower(
                        images.to(device=vision_model.device, dtype=vision_model.dtype),
                        output_hidden_states=True
                  )
                  image_features = vision_model.feature_select(image_forward_outs).to(images.dtype)

            return image_features

      vision_model.forward = forward_without_no_grad
      image_encoder = lambda x: vision_model(x)

      # delete the model (including LLM) to save memory
      del model
      gc.collect(); torch.cuda.empty_cache()

      diff_aug = None

      img_size = 336

      return image_encoder, image_processor, diff_aug, img_size

# def get_image_encoder_llava():
#       '''
#       Return: the image encoder, image processor and the data augmention used during training

#       image_processor is only for sanity check in test_attack_efficacy()
#       diff_aug will be used in crafting adversarial examples
#       '''
#       model_path = "liuhaotian/llava-v1.5-7b"
#       tokenizer, model, image_processor, context_len = load_pretrained_model(
#       model_path=model_path,
#       model_base=None,
#       model_name=get_model_name_from_path(model_path)
#       )

#       vision_model = copy.deepcopy(model.model.vision_tower); vision_model.eval()
#       # In llava, the forward function of CLIP is wrapped with torch.no_grad, which we get rid of below
#       image_encoder_ = vision_model.forward.__wrapped__
#       image_encoder = lambda x: image_encoder_(x)

#       # delete the model (including LLM) to save memory
#       del model
#       gc.collect(); torch.cuda.empty_cache()

#       diff_aug = None

#       img_size = 336

#       return image_encoder, image_processor, diff_aug, img_size

def encode_image_llava(image_encoder, image_processor, X_adv, img_size, bs, diff_aug, orig_sizes, normalize=normalize):
#     image_processor = CLIPImageProcessor(
#         size=img_size,  # Resizing target
#         resample="bicubic",  # Bicubic interpolation
#         do_center_crop=True,  # Center crop to img_size
#         image_mean=[0.48145466, 0.4578275, 0.40821073],  # Mean for normalization
#         image_std=[0.26862954, 0.26130258, 0.27577711]  # Std for normalization
#     )
    X_adv_resized = image_processor(X_adv).unsqueeze(0)
#     X_adv_resized = torch.cat([
#     TF.resize(
#             X_adv[j][:, :orig_sizes[j][1], :orig_sizes[j][0]], 
#             (img_size, img_size), 
#             interpolation=InterpolationMode.BICUBIC
#             ).unsqueeze(0) for j in range(bs)
#     ], dim=0)

#     if diff_aug is not None:
#             # NOTE: using differentiable randomresizedcrop here 
#             X_adv_input_to_model = normalize(diff_aug(X_adv_resized))
#     else:
#             X_adv_input_to_model = normalize(X_adv_resized)
    
    return image_encoder(X_adv_input_to_model)

# def get_response_llava(image, text_prompt, tokenizer, model, image_processor, conv, args):
#       conv = copy.deepcopy(conv)
#       conv.messages = [] # reset

#       image_tensor = process_images([image], image_processor, model.config)
#       if type(image_tensor) is list:
#             image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
#       else:
#             image_tensor = image_tensor.to(model.device, dtype=torch.float16)

#       inp = text_prompt
#       if model.config.mm_use_im_start_end:
#             inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
#       else:
#             inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
#       conv.append_message(conv.roles[0], inp)
#       conv.append_message(conv.roles[1], None)
#       prompt = conv.get_prompt()

#       input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
#       stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#       keywords = [stop_str]
#       stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

#       with torch.inference_mode():
#             output_ids = model.generate(
#                   input_ids,
#                   images=image_tensor,
#                   do_sample=True if args.temperature > 0 else False,
#                   temperature=args.temperature,
#                   max_new_tokens=args.max_new_tokens,
#                   streamer= None, # streamer, 
#                   use_cache=True,
#                   stopping_criteria=[stopping_criteria])
            
#       outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

#       return outputs

def get_llava_model(args):
      # if 'liuhaotian' in args.model_path:
      #       args.model_base = None
      # print(f'Loading {args.model_path}')
      # model_name = 'llava_v1.5_lora' # NOTE: we assume that the model checkpoint is lora; see "load_pretrained_model" function in llava for more details.
      # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

      ### Eval without 
      model_path = "liuhaotian/llava-v1.5-7b"
      model_name = get_model_name_from_path(model_path)
      tokenizer, model, image_processor, context_len = load_pretrained_model(
      model_path=model_path,
      model_base=None,
      model_name=model_name
      )

      if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
      elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
      elif "mpt" in model_name.lower():
            conv_mode = "mpt"
      else:
            conv_mode = "llava_v0"

      conv = conv_templates[conv_mode].copy()

      return tokenizer, model, image_processor, conv

# def get_response_llava(image, text_prompt, tokenizer, model, image_processor, conv, args):
#       conv = copy.deepcopy(conv)
#       conv.messages = [] # reset

#       image_tensor = process_images([image], image_processor, model.config)
#       if type(image_tensor) is list:
#             image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
#       else:
#             image_tensor = image_tensor.to(model.device, dtype=torch.float16)

#       inp = text_prompt
#       if model.config.mm_use_im_start_end:
#             inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
#       else:
#             inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
#       conv.append_message(conv.roles[0], inp)
#       conv.append_message(conv.roles[1], None)
#       prompt = conv.get_prompt()

#       input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
#       stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#       keywords = [stop_str]
#       stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

#       with torch.inference_mode():
#             output_ids = model.generate(
#                   input_ids,
#                   images=image_tensor,
#                   do_sample=True if args.temperature > 0 else False,
#                   temperature=args.temperature,
#                   max_new_tokens=args.max_new_tokens,
#                   streamer= None, # streamer, 
#                   use_cache=True,
#                   stopping_criteria=[stopping_criteria])
            
#       outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

#       return outputs

def get_response_llava(image, text_prompt, tokenizer, model, image_processor, conv, args):
      conv = copy.deepcopy(conv)
      conv.messages = [] # reset

      image_tensor = process_images([image], image_processor, model.config)
      if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
      else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

      inp = text_prompt
      if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
      else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
      conv.append_message(conv.roles[0], inp)
      conv.append_message(conv.roles[1], None)
      prompt = conv.get_prompt()

      input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
      stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
      keywords = [stop_str]
      stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

      with torch.inference_mode():
            output_ids = model.generate(
                  input_ids,
                  images=image_tensor,
                  do_sample=True if args.temperature > 0 else False,
                  temperature=args.temperature,
                  max_new_tokens=args.max_new_tokens,
                  streamer= None, # streamer, 
                  use_cache=True,
                  stopping_criteria=[stopping_criteria])
            
      outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

      return outputs