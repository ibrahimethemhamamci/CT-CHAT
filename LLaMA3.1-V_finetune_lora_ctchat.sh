
# Load necessary modules
module load anaconda3
module load a100
module load cuda/12.4.1
source activate llava_new


deepspeed --master_port 12540  llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --version llama3_1 \
    --data_path path_to_train_json.json \
    --image_folder path_to_train_encodings/encodings/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type "attn_pool+mlp2x_gelu" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --pretrain_mm_mlp_adapter ./checkpoints_new/llava-llama3.1_8B_ctclip-pretrain_256attention/mm_projector.bin \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints_new/llava-llama3.1_8B_ctclip-finetune_256-lora \
    --num_train_epochs 10 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 12 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 10000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 128000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --lazy_preprocess True \
    --report_to wandb\
    --deepspeed zero3.json
    

