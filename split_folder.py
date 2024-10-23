import os
import shutil
import random

def split_dataset(folder_path, train_ratio=0.8, output_dir='./output'):
    """
    Split a folder of images into train and test sets.
    
    Args:
    folder_path (str): Path to the folder containing images.
    train_ratio (float): Proportion of images to use for training (default is 0.8).
    output_dir (str): Directory where train/test folders will be saved.
    
    Returns:
    None
    """

    # Get list of image files in the folder (supports .jpg, .jpeg, .png)
    valid_extensions = ['.jpg', '.jpeg', '.png']
    images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]

    # Shuffle images
    random.shuffle(images)

    # Split into train and test
    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    test_images = images[train_size:]

    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy images to train and test directories
    for img in train_images:
        shutil.copy(os.path.join(folder_path, img), os.path.join(train_dir, img))

    for img in test_images:
        shutil.copy(os.path.join(folder_path, img), os.path.join(test_dir, img))

    print(f"Train/Test split completed. {len(train_images)} images in 'train' and {len(test_images)} images in 'test'.")

if __name__ == "__main__":
    folder_path = "path/to/your/images"  # Replace with the actual folder containing images
    split_dataset(folder_path, train_ratio=0.8)