python sample_fs.py \
  --ckpt_dir="ckpt/" \
  --character_input \
  --content_character="隆" \
  --style_image_dir="/home/yamamoto/workspace/font-dataset/refs/Kaisei_Tokumin_Medium/eight-shot" \
  --max_style_refs=8 \
  --save_image \
  --save_image_dir="outputs/" \
  --device="cuda:0" \
  --algorithm_type="dpmsolver++" \
  --guidance_type="classifier-free" 
# --guidance_scale=7.5 \
# --num_inference_steps=20 \
# --method="multistep"
#   --content_image_path="data_examples/sampling/example_content.jpg" \
