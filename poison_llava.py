import os
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from torchvision.utils import save_image

from models import get_image_encoder_llava, encode_image_llava, get_image_encoder_internlm, encode_image_internlm
from datasets import collate_fn, SingleTargetPairedImageDataset
from poison_utils import load_image, load_image_tensors, L2_norm

# diff augmentation
# import kornia
from augmentation_zoo import *

def parse_args():
    parser = argparse.ArgumentParser(description="Poisoning")

    parser.add_argument("--task_data_pth", default='data/task_data/Biden_base_Trump_target', help='task_data_pth folder contains base_train and target_train folders for constructing poison images') 
    parser.add_argument("--poison_save_pth", default='data/poisons/llava/Biden_base_Trump_target', help='Output path for saving pure poison images & original captions') 

    parser.add_argument("--iter_attack", type=int, default=4000)
    parser.add_argument("--lr_attack", type=float, default=0.2)

    parser.add_argument("--diff_aug_specify", type=str, default=None, help='if None, using the default diff_aug of the VLM')

    parser.add_argument("--batch_size", type=int, default=60, help='batch size for running the PGD attack. Modify it according to your GPU memory') 

    args = parser.parse_args()

    if args.diff_aug_specify == "None":
      args.diff_aug_specify = None

    return args

def embedding_attack_Linf(encode_images, image_base, image_victim, emb_dist, masks, orig_sizes, \
                     iters=100, lr=1/255, eps=8/255, diff_aug=None, resume_X_adv=None):
      '''
      optimizing x_adv to minimize emb_dist( img_embed of x_adv, img_embed of image_victim ) within Lp constraint using PGD

      image_encoder: the image embedding function (e.g. CLIP, EVA)
      image_base, image_victim: images BEFORE normalization, between [0,1]
      emb_dist: the distance metrics for vision embedding (such as L2): take a batch of bs image pairs as input, \
            and output EACH of pair-wise distances of the whole batch (size = [bs])

      eps: for Lp constraint
      lr: the step size. The update is grad.sign * lr
      diff_aug: using differentiable augmentation, e.g. RandomResizeCrop
      resume_X_adv: None or an initialization for X_adv

      return: X_adv between [0,1]
      '''
      # assert len(image_base.size()) == len(image_victim.size()) and len(image_base.size()) == 4, 'image size length should be 4'
      # assert image_base.size(0) == image_victim.size(0), 'image_base and image_victim contain different number of images'
      bs = image_base.size(0)
      device = image_base.device

      with torch.no_grad():
            embedding_targets = encode_images(image_victim, orig_sizes)

      X_adv = image_base.clone().detach() + (torch.rand(*image_base.shape)*2*eps-eps).to(device)
      if resume_X_adv is not None:
            print('Resuming from a given X_adv')
            X_adv = resume_X_adv.clone().detach()
      X_adv.data = X_adv.data.clamp(0,1)
      X_adv.requires_grad_(True) 

      optimizer = optim.SGD([X_adv], lr=lr)
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(iters*0.5)], gamma=0.5)

      loss_best = 1e8 * torch.ones(bs).to(device)
      X_adv_best = resume_X_adv.clone().detach() if resume_X_adv is not None else torch.rand(*image_base.shape).to(device)

      for i in tqdm(range(iters)):
      # for i in range(iters):
            embs = encode_images(X_adv, orig_sizes)

            loss = emb_dist(embs, embedding_targets) # length = bs

            if i% max(int(iters/1000),1) == 0:
                  if (loss < loss_best).sum()>0:
                        index = torch.where(loss < loss_best)[0]
                        loss_best[index] = loss.clone().detach()[index].to(loss_best[index].dtype)
                        X_adv_best[index] = X_adv.clone().detach()[index]

            loss = loss.sum() 
            optimizer.zero_grad()
            loss.backward()

            if i% max(int(iters/20),1) == 0:
                  print('Iter :{} loss:{:.4f}, lr * 255:{:.4f}'.format(i,loss.item()/bs, scheduler.get_last_lr()[0]*255))

            for j in range(bs):
                  X_adv.grad[j] *= (1 - masks[j])

            # Linf sign update
            X_adv.grad = torch.sign(X_adv.grad)
            optimizer.step()
            scheduler.step()
            X_adv.data = torch.minimum(torch.maximum(X_adv, image_base - eps), image_base + eps) 
            X_adv.data = X_adv.data.clamp(0,1)     
            X_adv.grad = None  

            if torch.isnan(loss):
                  print('Encounter nan loss at iteration {}'.format(i))
                  break                 

      with torch.no_grad():
            embs = encode_images(X_adv_best, orig_sizes)
            loss = emb_dist(embs, embedding_targets)
            # print('Best Total loss vector:{}'.format(loss))
            print('Best Total loss:{:.4f}'.format(loss.mean().item()))

      return X_adv_best, loss.detach()

if __name__ == "__main__":
      args = parse_args()

      # if os.path.exists(args.poison_save_pth):
      #       raise ValueError('{} already exists for saving pure poisoned data. Delete it or choose another path!'.format(args.poison_save_pth))
      # else:
      #       os.makedirs(os.path.join(args.poison_save_pth,'image')) 
      if not os.path.exists(args.poison_save_pth):
        os.makedirs(os.path.join(args.poison_save_pth, 'image')) # mathvista edit
      print(f'Poisong images will be saved to {args.poison_save_pth}')
      print(f'iter_attack {args.iter_attack}, lr_attack {args.lr_attack}')
            

      ###### model preparation ######
      image_encoder, image_processor, diff_aug, img_size = get_image_encoder_internlm()
      encode_images = lambda imgs, orig_sizes: encode_image_internlm(image_encoder, imgs, img_size, args.batch_size, diff_aug, orig_sizes) 

      if args.diff_aug_specify is not None:
            diff_aug = get_image_augmentation(augmentation_name=args.diff_aug_specify, image_size=img_size)
      else:
            print('Using default diff_aug')

      ###### data preparation ######
      images_base, image_target = load_image_tensors(args.task_data_pth,img_size)
      dataset_pair = SingleTargetPairedImageDataset(images_base=images_base, image_target=image_target)



      # dataloader_pair = torch.utils.data.DataLoader(dataset_pair, batch_size=args.batch_size, shuffle=False)
      dataloader_pair = torch.utils.data.DataLoader(
            dataset_pair, batch_size=args.batch_size, 
            shuffle=False, collate_fn=collate_fn  
      )

      ###### Resume by checking already saved images ######
      # Get the set of already saved poisoned images to skip
      saved_images = set(os.listdir(os.path.join(args.poison_save_pth, 'image')))
      saved_image_ids = {int(fname.split('.')[0]) for fname in saved_images if fname.endswith('.jpg')}  # assuming saved as .jpg


      ###### Running attack optimization ######
      X_adv_list = []
      loss_attack_list = []
      start_idx = 0
      end_idx = 0
      for i, (image_base, image_victim, masks, original_sizes) in  tqdm(enumerate(dataloader_pair), total=len(dataloader_pair), desc="Processing Batches"):
            # if i == 1:
            #       raise Exception("Stopping on purpose here")

            start_idx = end_idx + 1
            end_idx = start_idx + image_base.size(0) - 1 # mathvista

            if all(idx in saved_image_ids for idx in range(start_idx, end_idx+1)):
                  print(f'Batch {i} already processed, skipping...')
                  continue

            print('batch_id = ',i)
            image_base, image_victim, masks = image_base.cuda(), image_victim.cuda(), masks.cuda()
            X_adv, loss_attack = embedding_attack_Linf(
                  encode_images=encode_images, image_base=image_base, 
                  image_victim=image_victim, emb_dist=L2_norm, 
                  iters=args.iter_attack, lr=args.lr_attack/255, 
                  eps=8/255, diff_aug=diff_aug, resume_X_adv=None, 
                  masks=masks, orig_sizes=original_sizes
            )
            
            ###### Save poisoned images after each batch ######
            for j in range(X_adv.size(0)):  # Save each image in the batch
                  img_idx = start_idx + j
                  img_pth = os.path.join(args.poison_save_pth, 'image', f'{img_idx}.jpg')
                  if img_idx not in saved_image_ids:  # Only save if it doesn't already exist
                        save_image(X_adv[j][:, :original_sizes[j][1], :original_sizes[j][0]].cpu(), img_pth)
                        print(f'Saved poisoned image {img_idx} to {img_pth}') # mathvista

            X_adv_list.append(X_adv)
            loss_attack_list.append(loss_attack)

      # X_adv = torch.cat(X_adv_list,axis=0)
      loss_attack = torch.cat(loss_attack_list,dim=0)

      # sanity check (taking into consideration of loading images and image processor)
    #   test_attack_efficacy(image_encoder=image_encoder, image_processor=image_processor, \
    #                  task_data_pth=args.task_data_pth, poison_data_pth=args.poison_save_pth, img_size=img_size, sample_num=50)
      print(f'Poisong images are saved to {args.poison_save_pth}')