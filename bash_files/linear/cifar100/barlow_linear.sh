python3 main_linear.py \
    --dataset cifar100 \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --train_dir cifar100/train \
    --val_dir cifar100/val \
    --split_strategy class \
    --num_tasks 20 \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.05 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 4 \
    --name TPFR-cls-20-tsks \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project capstone-design-linear-eval-20tsks \
    --entity kjk5916-yonsei-university \
    --wandb \
    --save_checkpoint
