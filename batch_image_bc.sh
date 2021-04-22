#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python batch_image_BC.py \
--policy 'StateImgBC' \
--env "FetchPickAndPlace-v1" \
--lr 0.003 \
# --log
# --policy 'StateImgToStateVectorBC' \
# --env "FetchReach-v1" \
