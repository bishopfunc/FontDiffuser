#!/bin/bash
char_list="人一日大年出本中子見国言上分生手自行者二間事思時気会十家女三前的方入小地合後目長場代私下立部学物月田何来彼話体動社知理山内同心発高実作当新世今書度明五戦力名金性対意用男主通関文屋感郎業定政持道外取所現"
exp_name="sample_content_image_100_ref_Chiron_Hei_HK_Bold"
for (( i=0; i<${#char_list}; i++ )); do
    char="${char_list:i:1}"
    echo "Sampling character: $char"
    python sample.py \
        --ckpt_dir="ckpt/" \
        --character_input \
        --content_character="$char" \
        --style_image_path="/home/yamamoto/workspace/font-dataset/data/freq-kanji-top500/Chiron_Hei_HK_Bold/images/愛.png" \
        --save_image \
        --save_image_dir="outputs/${exp_name}/" \
        --device="cuda:0" \
        --algorithm_type="dpmsolver++" \
        --guidance_type="classifier-free" 
done

