import os
import re
import torch
import json
import shutil
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode

def findWholeWord(w):
    '''
    Checking if a sentence contains a word (not a substring).

    Example: 
    findWholeWord('seek')('those who seek shall find') is not None
    findWholeWord('seek')('those who shall find') is None
    '''
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def save_poison_data(images_to_save, caption_pth, save_path, orig_sizes):
      '''
      Save the pure poison data set as the same folder format as cc_sbu_align

      Input:
      images_to_save: a batch of image tensors (perturbed base_train images)
      caption_pth: json file path of captions for the unpoisoned images (base_train captions)
      save_path: path for saving poisoned images and original captions. 
      need to save to png, not jpeg.
      '''
      # assert len(images_to_save.size()) == 4, 'images_to_save should be a batch of image tensors, 4 dimension'
      assert len(images_to_save[0].size()) == 3, 'images_to_save should be a batch of image tensors, 3 dimension each'

      with open(caption_pth) as file:    
            cap = json.load(file)
      num_total = len(cap['annotations'])
      # assert images_to_save.size(0) == num_total, f'numbers of images and captions are different - images_to_save.size(0): {images_to_save.size(0)} num_total: {num_total}'
      assert len(images_to_save) == num_total, f'numbers of images and captions are different - images_to_save.size(0): {len(images_to_save)} num_total: {num_total}'

      # save image using the original image_id
      for i in range(num_total):
            image_id = cap['annotations'][i]['image_id']
            img_pth = os.path.join(save_path, 'image', '{}.jpg'.format(image_id)) # for mathvista
            # img_pth = os.path.join(save_path, 'image', '{}.png'.format(image_id))
            # save_image(images_to_save[i],img_pth)
            img = images_to_save[i]
            idx_1 = orig_sizes[i][1]
            idx_2 = orig_sizes[i][0]
            img = img[:, :idx_1, :idx_2].cpu()
            save_image(images_to_save[i][:, :orig_sizes[i][1], :orig_sizes[i][0]].cpu(), img_pth)

            # rename to .jpg
            img_pth_jpg = os.path.join(save_path, 'image', '{}.jpg'.format(image_id))
            os.rename(img_pth,img_pth_jpg)

      # copy the json file
      shutil.copyfile(caption_pth, os.path.join(save_path,'cap.json'))

      print('Finished saving the pure poison data to {}'.format(save_path))

def L2_norm(a,b):
      '''
      a,b: batched image/representation tensors
      '''
      assert a.size(0) == b.size(0), 'two inputs contain different number of examples'
      bs = a.size(0)

      dist_vec = (a-b).view(bs,-1).norm(p=2, dim=1)

      return dist_vec

def load_image(image_file, show_image=False):
      if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
      else:
            image = Image.open(image_file).convert('RGB')
      if show_image:
            plt.imshow(image)
            plt.show()
      return image

def load_image_tensors(task_data_pth,img_size,training_mode):
    '''
    Input:
    task_data_pth needs to contain two subfolders: base_train and target_train;
    task_data_pth/base_train needs to contain cap.json caption file

    img_size is the size of image for the VLM model. used for resizing.

    Output: list of image tensors from base_train, target_train 
    '''
    with open(os.path.join(task_data_pth,'base_train','cap.json')) as file:    
        base_train_cap = json.load(file)
    num_total = len(base_train_cap['annotations'])

    images_target = []
    images_base = []
    
    if training_mode == "paired":
        resize_fn = transforms.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.BICUBIC
                )

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
    elif training_mode == "single_target":
        resize_fn = lambda x: x # identity

        for i in tqdm(range(num_total), desc="Loading image tensors"):
            image_id = base_train_cap['annotations'][i]['image_id']

            if "MathVista" in task_data_pth:
                  image_base_pth = os.path.join(task_data_pth, 'base_train', f'{image_id}.jpg') #jpg for mathvista
            else:
                  image_base_pth = os.path.join(task_data_pth, 'base_train', f'{image_id}.png') 

            images_base.append(transforms.ToTensor()(resize_fn(load_image(image_base_pth))).unsqueeze(0)) 


            #   images_target.append(transforms.ToTensor()(resize_fn(load_image(image_target_pth))).unsqueeze(0)) 

        # mathvista
        image_target_pth = os.path.join(task_data_pth, 'target_train', '0.png')
        image_target = transforms.ToTensor()(resize_fn(load_image(image_target_pth))).unsqueeze(0)

        print(f'Finishing loading {num_total} pairs of base and target images for poisoning, size={len(images_base)}')

        return images_base, image_target
    elif training_mode == "all_target":
        resize_fn = lambda x: x # identity

        with open(os.path.join(task_data_pth,'target_train','cap.json')) as file:    
            target_train_cap = json.load(file)
        num_total_target = len(target_train_cap['annotations'])

        for i in tqdm(range(num_total), desc="Loading image tensors"):
            image_id = base_train_cap['annotations'][i]['image_id']

            image_base_pth = os.path.join(task_data_pth, 'base_train', f'{image_id}.jpg') #jpg for mathvista

            images_base.append(transforms.ToTensor()(resize_fn(load_image(image_base_pth))).unsqueeze(0)) 
      
        for i in tqdm(range(num_total_target), desc="Loading image tensors"):
            image_id = target_train_cap['annotations'][i]['name']

            image_base_pth = os.path.join(task_data_pth, 'target_train', f'{image_id}.png') 

            images_target.append(transforms.ToTensor()(resize_fn(load_image(image_base_pth))).unsqueeze(0)) 

        print(f'Finishing loading {num_total} pairs of base and target images for poisoning, size={len(images_base)}')

        return images_base, images_target, target_train_cap["annotations"]
    else:
        raise Exception("Please provide valid training mode")

      



    

def test_attack_efficacy(image_encoder, image_processor, task_data_pth, poison_data_pth, img_size, sample_num=20):
      '''
      Sanity check after crafting poison model
      
      Reload image_base, image_target and image_poison from jpg
      Go through image processor, and check the relative distance in the image embedding space
      sample_num: only compute statistics for the first sample_num image triples and then take the average

      Output: will print averaged latent_dist(image_base,image_target) and latent_dist(image_poison,image_target)
      also output the pixel distance between base and poison images

      NOTE: image_processor includes data augmentation. However, when using differantial jpeg during creating poison image,
      the image_processor will not include jpeg operation. 
      '''
      # RGB image
      images_base, images_target = [], []
      images_poison = []

      # load data
      with open(os.path.join(poison_data_pth,'cap.json')) as file:    
            cap = json.load(file)
      num_total = len(cap['annotations'])

      for i in range(num_total):
            image_id = cap['annotations'][i]['image_id']

            # image_base_pth = os.path.join(task_data_pth, 'base_train', f'{image_id}.png')
            if "MathVista" in task_data_pth:
                  image_target_pth = os.path.join(task_data_pth, 'target_train', f'{0}.png')
                  image_base_pth = os.path.join(task_data_pth, 'base_train', f'{image_id}.jpg') # jpg for mathvista
            else:
                  image_base_pth = os.path.join(task_data_pth, 'base_train', f'{image_id}.png') # png for original
                  image_target_pth = os.path.join(task_data_pth, 'target_train', f'{image_id}.png')
                  
            image_poison_pth = os.path.join(poison_data_pth, 'image', f'{image_id}.jpg')

            images_base.append((load_image(image_base_pth)))
            images_target.append((load_image(image_target_pth)))
            images_poison.append((load_image(image_poison_pth)))

            if i >= sample_num:
                  break

      resize_fn = transforms.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.BICUBIC
                )

      # compute embedding distance
      dist_base_target_list = []
      dist_poison_target_list = []
      pixel_dist_base_poison = [] # Linf distance in pixel space
      for i in range(len(images_base)):
            image_base, image_target, image_poison = images_base[i], images_target[i], images_poison[i]

            emb_base = image_encoder( torch.from_numpy(image_processor(image_base)['pixel_values'][0]).cuda().unsqueeze(0) )
            emb_target = image_encoder( torch.from_numpy(image_processor(image_target)['pixel_values'][0]).cuda().unsqueeze(0) )
            emb_poison = image_encoder( torch.from_numpy(image_processor(image_poison)['pixel_values'][0]).cuda().unsqueeze(0) )

            dist_base_target_list.append( (emb_base - emb_target).norm().item() )
            dist_poison_target_list.append( (emb_poison - emb_target).norm().item() )
            pixel_dist_base_poison.append( torch.norm(transforms.ToTensor()(image_base) - transforms.ToTensor()(image_poison), float('inf')) )

      dist_base_target_list = torch.Tensor(dist_base_target_list)
      dist_poison_target_list = torch.Tensor(dist_poison_target_list)
      pixel_dist_base_poison = torch.Tensor(pixel_dist_base_poison)

      print('\n Sanity check of the optimization, considering image loading and image processor')
      print(f'>>> ratio betwen dist_base_target and dist_poison_target:\n{dist_base_target_list/dist_poison_target_list}')
      print(f'ratio mean: {(dist_base_target_list/dist_poison_target_list).mean()}')
      print(f'>>> Max Linf pixel distance * 255 between base and poison: {(pixel_dist_base_poison*255).max()}')
      print(f'>>> Argmax Linf pixel distance * 255 between base and poison: {torch.argmax(pixel_dist_base_poison*255)}')

      return 