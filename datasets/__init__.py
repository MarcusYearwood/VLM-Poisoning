import torch.nn.functional as F
import random
import torch
import torchvision 
from PIL import Image
import numpy as np

class PairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_base, images_target):
        '''
        both input image are tensors with (num_example, 3, h, w)
        This dataset be used to construct dataloader for batching
        '''
        super().__init__()
        assert images_base.shape[0] == images_target.shape[0]
        self.images_base = images_base
        self.images_target = images_target

    def __len__(self):
        return self.images_base.shape[0]

    def __getitem__(self, index):
        return self.images_base[index].unsqueeze(0), self.images_target[index].unsqueeze(0)

class RandomPairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_base, images_target):
        '''
        both input image are tensors with (num_example, 3, h, w)
        This dataset be used to construct dataloader for batching
        '''
        super().__init__()
        self.images_base = images_base
        self.images_target = images_target

    def __len__(self):
        return self.images_base.shape[0]

    def __getitem__(self, index):
        target_index = random.randint(0, len(self.images_target)-1)
        return self.images_base[index], self.images_target[target_index]

class SingleTargetPairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_base, image_target):
        '''
        both input image are tensors with (num_example, 3, h, w)
        This dataset be used to construct dataloader for batching
        '''
        super().__init__()
        self.images_base = images_base
        self.image_target = image_target

    def __len__(self):
        return len(self.images_base)

    def __getitem__(self, index):
        return self.images_base[index], self.image_target

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_caps):
        '''
        image_caps is list of dicts with image paths
        '''
        super().__init__()
        self.image_caps = image_caps

    def __len__(self):
        return len(self.image_caps)

    def __getitem__(self, index):
        '''
        returns torch image with values 0 to 255 with shape (3, height, width)
        '''
        with Image.open(self.image_caps[index]["path"]) as img:
            pil_image = img
            img_tensor = torchvision.transforms.PILToTensor()(img.convert('RGB'))
            return img_tensor, self.image_caps[index]


def collate_fn(batch):
    images_base, images_target = zip(*batch)

    # Find the maximum height and width in the batch for padding
    max_height = max(img.size(2) for img in images_base)
    max_width = max(img.size(3) for img in images_base)


    # Create lists to store padded images and masks
    padded_images_base = []
    original_sizes = []
    masks = []

    for img in images_base:
        _, __, h, w = img.shape  # Channels, Height, Width
        original_sizes.append((w, h))  # Store original size (width, height)

        # Calculate padding (left, right, top, bottom)
        padding = (0, max_width - w, 0, max_height - h)

        # Pad the image
        padded_img = F.pad(img, padding, value=0)  # Zero-padding
        padded_images_base.append(padded_img)

        # Create a padding mask (1 for padded, 0 for original)
        mask = torch.ones(1, max_height, max_width)  # Initialize mask with ones
        mask[:, :h, :w] = 0  # Set original area to 0 (no padding)
        masks.append(mask)
    
    images_target = torch.cat(images_target, dim=0)
    images_base = torch.cat(padded_images_base, dim=0)
    masks = torch.cat(masks, dim=0)

    return images_base, images_target, masks, original_sizes

def collate_fn_image(batch):
    '''
    returns images of shape (1,3,height,width)

    sample caps element:
    {
        "path": "data/mini_MathVista_grid/base/85.jpg",
        "pid": "0",
        "name": "85",
        "caption": "The graph illustrates the average weekly work hours for women aged 15 and older in Romania, Portugal, and Switzerland from 1995 to 2007, showing a noticeable divergence in work hours among these",
        "description": "The image is a graph titled \"Average usual weekly hours worked, women 15 years and older, 1995 to 2007.\" It illustrates the trends in average weekly work hours for women aged 15 and above in three countries: Romania, Portugal, and Switzerland.\n\n### Key Elements of the Image:\n\n1. **Axes:**\n   - The **horizontal axis** represents the years from 1995 to 2007.\n   - The **vertical axis** shows the average weekly hours worked, measured in increments from 0 to 40 hours.\n\n2. **Lines:**\n   - There are three lines representing each country:\n     - **Romania:** Displayed in **cyan**, this line shows some fluctuations, peaking around "
    }
    '''
    image, caps = zip(*batch)
    image = torch.stack(image, dim=0)
    bsz, c, h, w = image.shape
    assert bsz == 1
    
    return image.to(torch.float32), caps[0]

transform_fn = torchvision.transforms.Compose(
        [
            # torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            # torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.Lambda(lambda img: to_tensor(img)),
            torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0)),
        ]
    )

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

class EnsembleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_caps):
        '''
        image_caps is list of dicts with image paths
        '''
        super().__init__()
        self.image_caps = image_caps

    def __len__(self):
        return len(self.image_caps)

    def __getitem__(self, index):
        '''
        returns torch image with values 0 to 255 with shape (3, height, width)
        '''
        with Image.open(self.image_caps[index]["path"]) as img:
            pil_image = img
            # img_tensor = torchvision.transforms.PILToTensor()(img.convert('RGB'))
            img_tensor = transform_fn(img.convert('RGB'))
            return img_tensor, self.image_caps[index]
