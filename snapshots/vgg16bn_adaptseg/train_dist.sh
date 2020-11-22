#!/bin/sh
EXP_DIR=$(dirname $0)
EXP=$(echo $EXP_DIR | cut -d/ -f 2)
GPUS=8
PORT=12345

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    train_gta2bdd_distr.py \
    --dist True \
    --launcher pytorch \
    --model DeeplabVGGBN \
    --num-steps 40000 \
    --num-steps-stop 40000 \
    --snapshot-dir ./snapshots/$EXP \
    --save-pred-every 1000 \
    --learning-rate 0.01 \
    --learning-rate-D 0.005 \
    --lambda-seg 0.0 \
    --lambda-adv-target1 0.001 \
    --tensorboard \
    --log-dir ./snapshots/$EXP \
    --port $PORT \
    2>&1 | tee ./snapshots/$EXP/log.txt
