import sys
import argparse
import os
import gc

import json
import shutil

import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision import transforms
import torch.optim as optim
from torchvision.utils import save_image
from PIL import Image
import copy
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as TF


def count_images_in_folder(folder_path):
    # Define valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    # Initialize a counter
    image_count = 0
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Get the file extension
        ext = os.path.splitext(filename)[1].lower()
        
        # Check if the file has a valid image extension
        if ext in valid_extensions:
            image_count += 1

    return image_count

def rename_images(folder_path = '/usr/xtmp/mxy/VLM-Poisoning/data/task_data/MathVista_base_hamburgerFries_target/target_train'):
    # Get a list of all .png files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # Sort the files in their current numerical order (assuming filenames are like '1.png', '2.png', etc.)
    files.sort(key=lambda f: int(f.split('.')[0]))  # Sort based on the numerical part of the filename

    # Rename the files sequentially starting from 0
    for idx, filename in enumerate(files):
        # Construct the new filename
        new_filename = f'{idx}.png'
        
        # Get the full paths for renaming
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed {filename} to {new_filename}')

    print("Renaming complete.")

# def mathvista_transform(img_size):
#     return transforms.Compose([
#         # Resize the image while keeping the aspect ratio (resize based on the shorter side)
#         transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        
#         # Add padding to make the image square, pad with white
#         transforms.Pad((0, 0, img_size, img_size), fill=(255, 255, 255)),
        
#         # Center-crop it to the square size in case padding exceeds
#         transforms.CenterCrop(img_size)
#     ]) # resize for mathvista

def load_image(image_path, show_image=True):
    img = Image.open(image_path).convert('RGB')
    if show_image:
        plt.imshow(img)
        plt.show()
    return img

class mathvista_transform:
    def __init__(self, img_size, fill_color=(255, 255, 255)):
        self.img_size = img_size
        self.fill_color = fill_color

    def __call__(self, img):
        # Get original image size
        width, height = img.size

        # Calculate padding to make the image square
        if width > height:
            padding_top = (width - height) // 2
            padding_bottom = width - height - padding_top
            padding_left = padding_right = 0
        else:
            padding_left = (height - width) // 2
            padding_right = height - width - padding_left
            padding_top = padding_bottom = 0

        # Pad the image to make it square
        img = TF.pad(img, (padding_left, padding_top, padding_right, padding_bottom), fill=self.fill_color)

        # Resize the image to the desired size using bicubic interpolation
        img = TF.resize(img, (self.img_size, self.img_size), interpolation=Image.BICUBIC)

        return img


def resize_images(folder_path='/usr/xtmp/mxy/VLM-Poisoning/data/task_data/MathVista_base_hamburgerFries_target/base_train', save_path="/usr/xtmp/mxy/VLM-Poisoning/images_for_viewing"):
    # Get a list of all .png files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # Sort the files in their current numerical order (assuming filenames are like '1.png', '2.png', etc.)
    files.sort(key=lambda f: int(f.split('.')[0]))  # Sort based on the numerical part of the filename

    resize_fn = mathvista_transform(336)

    # Rename the files sequentially starting from 0
    for idx, filename in enumerate(files):
        # Construct the new filename
        
        # Get the full paths for renaming
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(save_path, filename)

        img = load_image(old_path)
        new_img = resize_fn(img)
        new_img.save(new_path)

        print(f'Resized {filename}')

    print("resizing complete.")

if __name__ == "__main__":
    resize_images()