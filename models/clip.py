import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from PIL import Image
import torchvision
from torch.nn import DataParallel
import torchvision.transforms as T
import warnings

d_type = torch.float16

def get_image_encoder_clip():
    vision_tower_name = "openai/clip-vit-large-patch14"
    warnings.filterwarnings("ignore")

    vision_tower = CLIPModel.from_pretrained(vision_tower_name).eval().cuda()
    # vision_tower.half()
    vision_tower.requires_grad_(True)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        vision_tower = vision_tower

    img_size = 224  # Standard image size for CLIP

    processor = CLIPTokenizer.from_pretrained(vision_tower_name)

    return vision_tower, processor, None, img_size

def encode_image_clip(image_encoder, X_adv, img_size, bs, diff_aug, orig_sizes):
    transform = T.Compose([
        T.Resize((img_size, img_size)),  # Resize image to CLIP's input size
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # CLIP normalization
    ])

    images = []
    for j in range(bs):
        orig_w, orig_h = orig_sizes[j]
        img = X_adv[j][:, :orig_h, :orig_w]

        if diff_aug:
            img = diff_aug(img).cuda()
        else:
            img = img.cuda()

        img = transform(img)
        images.append(img)

    # Concatenate images into a single tensor for processing
    images = torch.stack(images).to("cuda")

    with torch.autocast(device_type='cuda', dtype=d_type):
        image_embeds = image_encoder.get_image_features(pixel_values=images)

    return image_embeds


def i2t_similarity_clip(image_encoder, processor, image_tensor, text):
    # Preprocess image and text

    transform = T.Compose([
        T.Resize((224, 224)),  # Resize image to CLIP's input size
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # CLIP normalization
    ])

    image = transform(image_tensor.squeeze(0)).unsqueeze(0)
    # image = T.ToPILImage()(image_tensor.squeeze(0))

    text_inputs = processor([text], return_tensors='pt', padding=True, truncation=True)

    # inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    # Get image and text embeddings
    # inputs = {key: value.cuda() for key, value in inputs.items()}
    outputs = image_encoder(input_ids=text_inputs['input_ids'].cuda(),
                attention_mask=text_inputs['attention_mask'].cuda(),
                pixel_values=image.cuda())

    similarity = outputs.logits_per_image
    return similarity
