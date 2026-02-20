#!/bin/bash
char_list="深度字体"


exp_name="sample_content_image_100_ref_Noto_Serif_TC_Bold_few_shot_logo"
for (( i=0; i<${#char_list}; i++ )); do
    char="${char_list:i:1}"
    echo "Sampling character: $char"
    python sample_fs.py \
    --ckpt_dir="ckpt/" \
    --character_input \
    --content_character="$char" \
    --style_image_dir="/home/yamamoto/workspace/font-dataset/refs/Noto_Serif_TC_Bold/eight-shot" \
    --save_image \
    --save_image_dir="outputs/${exp_name}/" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" 
done