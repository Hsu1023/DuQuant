#!/bin/bash

python main.py \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits 4 \
    --abits 4 \
    --model meta-llama/Llama-2-7b-hf \
    --lwc \
    --alpha 0.6 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --eval_ppl \
    --task arc_easy,arc_challenge,hellaswag,winogrande,boolq,piqa\
