from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings


# build gpt class
class Llava:
    def __init__(self, pretrained="liuhaotian/llava-v1.5-13b", model_name="llava_qwen", patience=1000000, sleep_time=0, d_type=torch.float16, max_tokens=4096, temperature=0.1):
        warnings.filterwarnings("ignore")
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, device_map="auto")  # Add any other thing you want to pass in llava_model_args
        self.model.eval()
        self.patience = patience
        self.sleep_time = sleep_time
        self.d_type = d_type
        self.device = 'cuda'
        self.max_tokens = max_tokens
        self.temperature = temperature

    def get_response(self, image_path, user_prompt):
        image = Image.open(image_path)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=self.d_type, device='cuda') for _image in image_tensor]

        query = DEFAULT_IMAGE_TOKEN + user_prompt

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt_query = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_query, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size]



        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=self.d_type):
                        cont = self.model.generate(
                                    input_ids,
                                    images=image_tensor,
                                    image_sizes=image_sizes,
                                    do_sample=False,
                                    temperature=self.temperature,
                                    max_new_tokens=self.max_tokens,
                                )

        response = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        prediction = response[0].strip()
        return prediction
        
#         while patience > 0:
#             patience -= 1
#             try:
#                 # print("self.model", self.model)
#                 response, his = model.chat(self.tokenizer, query, image_path, do_sample=False, num_beams=3, use_meta=True)
#                 if self.n == 1:
#                     prediction = response.strip()
#                     if prediction != "" and prediction != None:
#                         return prediction
#                 else:
#                     prediction = [choice['message']['content'].strip() for choice in response['choices']]
#                     if prediction[0] != "" and prediction[0] != None:
#                         return prediction
#                     return prediction
                        
#             except Exception as e:
#                 print(e)
#                 # if "limit" not in str(e):
#                 #     print(e)
#                 # if "Please reduce the length of the messages." in str(e):
#                 #     print("!!Reduce user_prompt to", user_prompt[:-1])
#                 #     return ""
#                 if self.sleep_time > 0:
#                     time.sleep(self.sleep_time)
#         return ""
