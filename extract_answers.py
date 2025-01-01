from datasets import PairedImageDataset
import os

def load_image_tensors(path, cap_path = None):
    '''
    Input:
    task_data_pth needs to contain two subfolders: base_train and target_train;
    task_data_pth/base_train needs to contain cap.json caption file

    img_size is the size of image for the VLM model. used for resizing.

    Output: list of image tensors from base_train, target_train 
    '''
    if cap_path:
        with open(cap_path) as file:    
            base_train_cap = json.load(file)
        num_total = len(base_train_cap['annotations'])
    else:
        base_train_cap = {"annotations": ["image"]}

    images_target = []
    images_base = []
    
    resize_fn = lambda x: x #identity

    for i in range(num_total):
            image_id = base_train_cap['annotations'][i]['image_id']
            image_base_pth = os.path.join(task_data_pth, 'base_train', f'{image_id}.png')
            image_target_pth = os.path.join(task_data_pth, 'target_train', f'{image_id}.png')

            images_base.append(transforms.ToTensor()(resize_fn(load_image(image_base_pth))).unsqueeze(0)) 
            images_target.append(transforms.ToTensor()(resize_fn(load_image(image_target_pth))).unsqueeze(0)) 

    images_base = torch.cat(images_base, axis=0)
    images_target = torch.cat(images_target, axis=0)
    print(f'Finishing loading {num_total} pairs of base and target images for poisoning, size={images_base.size()}')

    return images_base, images_target

task = "Mini_MathVista_base_hamburgerFries_target"
poison_data_pth = os.path.join("data/poisons", task)
task_data_pth = os.path.join("data/task_data", task)

adv_images = load_image_tensors()

