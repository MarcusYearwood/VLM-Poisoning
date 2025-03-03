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
