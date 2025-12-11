CUDA_VISIBLE_DEVICES=0  /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ABO --output_dir datasets/ABO --rank 16 --world_size 24 &
CUDA_VISIBLE_DEVICES=1 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ABO --output_dir datasets/ABO  --rank 17 --world_size 24 &  
CUDA_VISIBLE_DEVICES=2 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ABO --output_dir datasets/ABO  --rank 18 --world_size 24 &
CUDA_VISIBLE_DEVICES=3 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ABO --output_dir datasets/ABO  --rank 19 --world_size 24 &
CUDA_VISIBLE_DEVICES=4 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ABO --output_dir datasets/ABO  --rank 20 --world_size 24 &
CUDA_VISIBLE_DEVICES=5 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ABO --output_dir datasets/ABO  --rank 21 --world_size 24 &
CUDA_VISIBLE_DEVICES=6 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ABO --output_dir datasets/ABO  --rank 22 --world_size 24 &
CUDA_VISIBLE_DEVICES=7 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ABO --output_dir datasets/ABO  --rank 23 --world_size 24 
wait


cd /shared/home/AZA0761/projects/TRELLIS &
conda activate trellis3 &
CUDA_VISIBLE_DEVICES=0 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse --rank 0 --world_size 8 &
CUDA_VISIBLE_DEVICES=1 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 1 --world_size 8 &  
CUDA_VISIBLE_DEVICES=2 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 2 --world_size 8 &
CUDA_VISIBLE_DEVICES=3 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 3 --world_size 8 &
CUDA_VISIBLE_DEVICES=4 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 4 --world_size 8 &
CUDA_VISIBLE_DEVICES=5 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 5 --world_size 8 &
CUDA_VISIBLE_DEVICES=6 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 6 --world_size 8 &
CUDA_VISIBLE_DEVICES=7 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 7 --world_size 8 
wait

ssh p4d-dy-a100-1
cd /shared/home/AZA0761/projects/TRELLIS 
conda activate trellis3 
СГ/shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ABO --output_dir datasets/ABO --rank 2 --world_size 3 --max_workers 8 &

wait

/shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/build_metadata.py ABO --output_dir datasets/ABO



rsync -av --progress /shared/home/AZA0761/projects/recgen/datasets/Objaverse_recgen/raw/ /shared/home/AZA0761/projects/TRELLIS/datasets/Objaverse/raw/ &
rsync -av --progress /shared/home/AZA0761/projects/recgen/datasets/Objaverse_recgen/latents/ /shared/home/AZA0761/projects/TRELLIS/datasets/Objaverse/latents/ &
rsync -av --progress /shared/home/AZA0761/projects/recgen/datasets/Objaverse_recgen/renders/*.json /shared/home/AZA0761/projects/TRELLIS/datasets/Objaverse/renders/*.json &
rsync -av --progress /shared/home/AZA0761/projects/recgen/datasets/Objaverse_recgen/voxels/ /shared/home/AZA0761/projects/TRELLIS/datasets/Objaverse/voxels/ &
rsync -av --progress /shared/home/AZA0761/projects/recgen/datasets/Objaverse_recgen/metadata.csv /shared/home/AZA0761/projects/TRELLIS/datasets/Objaverse/metadata.csv 
wait

CUDA_VISIBLE_DEVICES=0 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse --rank 0 --world_size 8 &
CUDA_VISIBLE_DEVICES=1 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 1 --world_size 8 &  
CUDA_VISIBLE_DEVICES=2 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 2 --world_size 8 &
CUDA_VISIBLE_DEVICES=3 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 3 --world_size 8 &
CUDA_VISIBLE_DEVICES=4 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 4 --world_size 8 &
CUDA_VISIBLE_DEVICES=5 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 5 --world_size 8 &
CUDA_VISIBLE_DEVICES=6 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 6 --world_size 8 &
CUDA_VISIBLE_DEVICES=7 /shared/home/AZA0761/.conda/envs/trellis3/bin/python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/Objaverse  --rank 7 --world_size 8 

/shared/home/AZA0761/.conda/envs/trellis3/bin/python train.py \
    --config configs/finetune_ss_flow_img_dit_L_16l8_fp16.json \
    --output_dir outputs/finetune_ss_flow_img_dit_L_16l8_fp16 \
    --data_dir datasets/Objaverse \
    --num_gpus 8

/shared/home/AZA0761/.conda/envs/trellis3/bin/python train.py \
    --config configs/experiments/abo_noisy_cond/finetune_ss_noisy_cond.json \
    --output_dir outputs/finetune_ss_flow_img_dit_L_16l8_fp16_final \
    --data_dir datasets/Objaverse \
    --num_gpus 8 \
    --use_wandb \
    --wandb_project noisy_trellis \
    --wandb_name finetune_ss_flow_img_dit_L_16l8_fp16_noisy

/shared/home/AZA0761/.conda/envs/trellis3/bin/python train.py \
    --config configs/experiments/abo_noisy_cond/finetune_slat_noisy_cond.json \
    --output_dir outputs/finetune_slat_flow_img_dit_L_64l8p2_fp16_noisy_abo \
    --data_dir datasets/Objaverse,datasets/ABO \
    --num_gpus 8 \
    --use_wandb \
    --wandb_project noisy_trellis \
    --wandb_name finetune_slat_flow_img_dit_L_64l8p2_fp16_noisy_abo

# Conservative finetuning (smaller LR, smaller noise, warmup, weight decay)
/shared/home/AZA0761/.conda/envs/trellis3/bin/python train.py \
    --config configs/experiments/abo_noisy_cond/finetune_slat_noisy_cond_conservative.json \
    --output_dir outputs/finetune_slat_flow_img_dit_L_64l8p2_fp16_noisy_abo_conservative \
    --data_dir datasets/Objaverse,datasets/ABO \
    --num_gpus 8 \
    --use_wandb \
    --wandb_project noisy_trellis \
    --wandb_name finetune_slat_noisy_conservative


python remove_checkpoints.py /shared/home/AZA0761/projects/TRELLIS/outputs/finetune_slat_flow_img_dit_L_64l8p2_fp16_noisy_abo/ckpts 15000 20000 25000 30000 35000 40000 45000 55000 60000 65000 70000 75000 80000 85000 90000 95000

python remove_checkpoints.py /shared/home/AZA0761/projects/TRELLIS/outputs/finetune_ss_flow_img_dit_L_16l8_fp16_final/ckpts 15000 20000 25000 30000 35000 40000 45000 55000 60000 65000 70000 75000 80000 85000 90000 95000


python train.py \
    --config configs/finetune_slat_flow_img_dit_L_64l8p2_fp16_lora.json \
    --output_dir outputs/slat_lora_combined \
    --data_dir /shared/home/AZA0761/projects/TRELLIS/datasets/ABO,/shared/home/AZA0761/projects/TRELLIS/datasets/Objaverse \
    --use_wandb \
    --wandb_project trellis-lora \
    --wandb_name slat_lora_combined2 \
    --num_gpus 4
