<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=8B
NUM_GPUS=2
=======
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=8B
NUM_GPUS=4
>>>>>>> recovery
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# echo "       --train_file /fs/scratch/PAS2138/roozbehn99/code4translation/finetuning/cleaned_dataset.jsonl \
# echo "    --train_file data/processed/tulu_v2/tulu_v2_data.jsonl "

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --use_flash_attn \
    --dataset_name allenai/tulu-3-sft-mixture \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir output/llama3_8b_finetuned \
    --with_tracking \
<<<<<<< HEAD
    --checkpointing_steps 250 \
    --report_to wandb \
    --wandb_entity roozbeh-n99 \
    --logging_steps 1
=======
    --checkpointing_steps 500 \
    --report_to wandb \
    --wandb_entity roozbeh-n99 \
    --logging_steps 1



# ------------------------
    # --model_name_or_path ../hf_llama3_models/${MODEL_SIZE} \
    # --tokenizer_name ../hf_llama3_models/${MODEL_SIZE} \
    # --train_file data/processed/tulu_v2/tulu_v2_data.jsonl \
>>>>>>> recovery
