import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import gather_object
from parallelformers import parallelize
import warnings
from PIL import Image
import torchvision

def get_image_encoder_internlm():
    vision_tower_name="internlm/internlm-xcomposer2d5-clip"
    d_type=torch.float16
    warnings.filterwarnings("ignore")

    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name).eval().cuda()
    vision_tower.requires_grad_(True)

    img_size=560

    return vision_tower, None, None, img_size

def encode_image_internlm(image_encoder, X_adv, img_size, bs, diff_aug, orig_sizes):
    images = []
    for j in range(bs):
        # Extract the original size of the image
        orig_w, orig_h = orig_sizes[j]

        # Crop the image to its original size before padding
        img = X_adv[j][:, :orig_h, :orig_w]

        # Apply your custom resizing logic with padding
        img = __resize_img__(img)

        # Resize to the target model size (560x560)
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode='bicubic')

        # Add normalization and augmentation (if any)
        if diff_aug:
            img = diff_aug(img).cuda()
        else:
            img = img.cuda()

        images.append(img)

    images = torch.cat(images, dim=0).cuda()

    # Encode the batch of images
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