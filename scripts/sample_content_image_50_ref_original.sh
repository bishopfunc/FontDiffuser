#!/bin/bash
char_list="人一日大年出本中子見国言上分生手自行者二間事思時気会十家女三前的方入小地合後目長場代私下立部学物月田"
exp_name="sample_content_image_50_ref_original"
for (( i=0; i<${#char_list}; i++ )); do
    char="${char_list:i:1}"
    echo "Sampling character: $char"
    python sample.py \
        --ckpt_dir="ckpt/" \
        --character_input \
        --content_character="$char" \
        --style_image_path="data_examples/sampling/example_style.jpg" \
        --save_image \
        --save_image_dir="outputs/${exp_name}/" \
        --device="cuda:0" \
        --algorithm_type="dpmsolver++" \
        --guidance_type="classifier-free" 
done

