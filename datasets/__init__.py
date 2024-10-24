import torch.nn.functional as F
import random
import torch

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
        return self.images_base[index], self.images_target[index]

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

def collate_fn(batch):
    # Extract the images from the batch
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