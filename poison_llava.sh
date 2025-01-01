#!/usr/bin/env bash 

### To run experiment, modify
model=llava #llava clip internlm
GPU_ID=0
task_name=Mini_MathVista_base_hamburgerFries_target # choose from: Biden_base_Trump_target, healthyFood_base_hamburgerFries_target, kidSports_base_kidVideoGame_target, lowFuelLight_base_engineLight_target, MathVista_base_hamburgerFries_target, Mini_MathVista_base_hamburgerFries_target

# standard iter_attack=4000
# CUDA_VISIBLE_DEVICES=$GPU_ID python poison_llava.py \
#  --task_data_pth data/task_data/$task_name --poison_save_pth data/poisons/llava/$task_name \
#  --iter_attack 1000 --lr_attack 0.2 --diff_aug_specify None --batch_size 20 \

 # training_modes: paired or single_target

python poison_llava.py \
 --task_data_pth data/task_data/$task_name --poison_save_pth ./data/poisons/$model/$task_name \
 --iter_attack 1000 --lr_attack 1 --diff_aug_specify None --batch_size 1 --model $model --training_mode single_target --eps 255