#!/bin/bash

python main.py \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 20 \
    --wbits 4 \
    --abits 4 \
    --model meta-llama/Llama-2-7b-hf \
    --lwc \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --eval_ppl \
    --task arc_easy,arc_challenge,hellaswag,winogrande,boolq,piqa
