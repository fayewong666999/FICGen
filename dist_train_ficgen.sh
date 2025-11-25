PY_ARGS=${@:1}
PORT=${PORT:-29499}

accelerate launch --main_process_port=29669 --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 train_ficgen.py \
    --pretrained_model_name_or_path=/cpfs/shared/public/mmc/stable-diffusion-v1-5/ \
    --train_data_dir=/cpfs/user/wenzhuangwang/FICGen/datasets/dior/train/ \
    --img_patch_path=/cpfs/user/wenzhuangwang/CC-Diff-CV/content/dior/ \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=40 \
    --allow_tf32 \
    --checkpointing_steps=600 \
    --num_train_epochs=300 \
    --learning_rate=1e-4 \
    --max_grad_norm=1 \
    --lr_scheduler=constant --lr_warmup_steps=0 \
    --output_dir=checkpoint/dior_sd15_new_up16_new \
    ${PY_ARGS}