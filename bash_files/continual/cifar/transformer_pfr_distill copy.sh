python3 main_continual.py \
    --dataset cifar100 \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --split_strategy class \
    --max_epochs 500 \
    --num_tasks 5 \
    --task_idx 1 \
    --gpus 0 \
    --num_workers 4 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.01 \
    --classifier_lr 0.1 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --name TPFR_cls \
    --project capstone-design \
    --entity kjk5916-yonsei-university \
    --wandb \
    --save_checkpoint \
    --method barlow_twins \
    --disable_knn_eval \
    --proj_hidden_dim 2048 \
    --output_dim 2048 \
    --scale_loss 0.1 \
    --distiller transformer_pfr \
    --distill_lamb 25 \
    --pretrained_model $PRETRAINED_PATH
