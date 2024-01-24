export CUDA_VISIBLE_DEVICES=0

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

export HDR_DATA_DIR="./hdr_roll"
export HFOV_DATA_DIR="./hfov60"

export OUTPUT_DIR="ThisIsTheFinal-lora-hdr-continuous-largeT@900/0_-5"
accelerate launch --main_process_port 11111 --num_processes 1 train_continuous_final.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --hdr_data_dir=$HDR_DATA_DIR \
  --hfov_data_dir=$HFOV_DATA_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a perfect mirrored reflective chrome ball sphere" \
  --instance_prompt_alternative="a perfect black dark mirrored reflective chrome ball sphere" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2500 \
  --seed="0" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --checkpointing_steps=250 \
  --max_negative_exposure=-5 \
  --dataloader_num_workers=2 \
  --timestep_sampler="largeT" \
  --start_timestep=900 \