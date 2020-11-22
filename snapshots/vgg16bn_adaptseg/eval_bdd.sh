#!/bin/sh
EXP_DIR=$(dirname $0)
size=960
epoch=30000
domain=rainy
#domain=snowy
#domain=cloudy
#domain=3domains
#domain=overcast

python -u -W ignore evaluate_bdd.py \
  --model DeeplabVGGBN \
  --data-list dataset/bdd_list/val/${domain}.txt \
  --restore-from ${EXP_DIR}/GTA5_${epoch}.pth \
  --save $EXP_DIR/results_${size}_${epoch}

python -u compute_iou.py \
    ./C-Driving/val/compound \
     ${EXP_DIR}/results_${size}_${epoch} \
    --devkit_dir dataset/bdd_list \
    --img_list dataset/bdd_list/val/${domain}.txt \
    --label_list dataset/bdd_list/val/${domain}_label.txt \
  2>&1 | tee ${EXP_DIR}/eval_${size}_${epoch}_${domain}.txt
