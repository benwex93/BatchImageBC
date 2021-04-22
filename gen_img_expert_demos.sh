#!/bin/bash

# Script to reproduce results

CUDA_VISIBLE_DEVICES=0 python gen_expert_dataset.py \
--env "FetchPickAndPlace-v1" \
--num_episodes 101 \
# --env "FetchReach-v1" \
