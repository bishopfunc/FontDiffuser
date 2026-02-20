import os
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from PIL import Image
from src import (
    FontDiffuserDPMPipelineFewShot,
    FontDiffuserModelDPMFewShot,
    build_content_encoder,
    build_ddpm_scheduler,
    build_style_encoder,
    build_unet,
)
from utils import (
    is_char_in_font,
    load_ttf,
    my_single_image,
    save_args_to_yaml,
    save_image_with_content_style,
    ttf2im,
)


def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--controlnet",
        type=bool,
        default=False,
        help="If in demo mode, the controlnet can be added.",
    )
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument(
        "--save_image_dir", type=str, default=None, help="The saving directory."
    )
    parser.add_argument(
        "--merge_style_refs",
        default=None,
        type=str,
        help="The method to merge multiple style refs. None or 'mean' or 'random_choice'.",
    )
    parser.add_argument("--style_image_dir", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")
    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def _collect_style_paths(args):
    paths = []

    style_image_dir = Path(args.style_image_dir)
    for img_path in style_image_dir.glob("*.png"):
        paths.append(img_path)
    return paths


def image_process(args, content_image=None, style_image=None):
    if args.character_input:
        assert args.content_character is not None, (
            "The content_character should not be None."
        )
        if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
            return None, None, None
        font = load_ttf(ttf_path=args.ttf_path)
        content_image = ttf2im(font=font, char=args.content_character)
        content_image_pil = content_image.copy()
    else:
        content_image = Image.open(args.content_image_path).convert("RGB")
        content_image_pil = None

    # ---- style (few-shot) ----
    style_paths = _collect_style_paths(args)
    assert len(style_paths) > 0, (
        f"No style reference images found in {args.style_image_dir}. Please provide at least one image."
    )
    print(f"Collected {len(style_paths)} style reference images: {style_paths}")
    style_images_pil = [Image.open(p).convert("RGB") for p in style_paths]

    # ---- transforms ----
    content_tf = transforms.Compose(
        [
            transforms.Resize(
                args.content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    style_tf = transforms.Compose(
        [
            transforms.Resize(
                args.style_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # content: [1, C, H, W]
    content_tensor = content_tf(content_image).unsqueeze(0)

    # style: [1, C, H, W] * N -> [N, C, H, W]
    style_tensor_stack = torch.stack([style_tf(im) for im in style_images_pil], dim=0)

    return content_tensor, style_tensor_stack, content_image_pil


def load_fontdiffuer_pipeline(args):
    # Load the model state_dict
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    model = FontDiffuserModelDPMFewShot(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    )
    model.to(args.device)
    print("Loaded the model state_dict successfully!")

    # Load the training ddpm_scheduler.
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler sucessfully!")

    # Load the DPM_Solver to generate the sample.
    pipe = FontDiffuserDPMPipelineFewShot(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded dpm_solver pipeline sucessfully!")

    return pipe


def sampling_fs(args, pipe):
    if not args.demo:
        os.makedirs(args.save_image_dir, exist_ok=True)
        # saving sampling config
        save_args_to_yaml(
            args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml"
        )

    if args.seed:
        set_seed(seed=args.seed)

    content_image_tensor, style_image_tensor_stack, content_image_pil = image_process(
        args=args
    )
    if args.save_image:
        style_paths = _collect_style_paths(args)
        for style_path in style_paths:
            image = Image.open(style_path).convert("RGB")
            my_single_image(
                save_dir=args.save_image_dir,
                image=image,
                save_name=f"ref_{style_path.stem}",
            )
    with torch.no_grad():
        content_image_tensor = content_image_tensor.to(args.device)
        style_image_tensor_stack = style_image_tensor_stack.to(args.device)
        print("Sampling by DPM-Solver++ ......")
        start = time.time()
        images = pipe.generate(
            content_images=content_image_tensor,
            style_images_stack=style_image_tensor_stack,  # updated
            batch_size=1,
            order=args.order,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=args.t_start,
            t_end=args.t_end,
            dm_size=args.content_image_size,
            algorithm_type=args.algorithm_type,
            skip_type=args.skip_type,
            method=args.method,
            correcting_x0_fn=args.correcting_x0_fn,
        )
        end = time.time()

        if args.save_image:
            print("Saving the image ......")
            if args.character_input:
                my_single_image(
                    save_dir=args.save_image_dir,
                    image=images[0],
                    save_name=f"gen_{args.content_character}",
                )
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=images[0],
                    content_image_pil=content_image_pil,
                    content_image_path=None,
                    style_image_path=args.style_image_path,
                    resolution=args.resolution,
                )
            else:
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=images[0],
                    content_image_pil=None,
                    content_image_path=args.content_image_path,
                    style_image_path=args.style_image_path,
                    resolution=args.resolution,
                )
            print(f"Finish the sampling process, costing time {end - start}s")
        return images[0]


if __name__ == "__main__":
    args = arg_parse()

    # load fontdiffuser pipeline
    pipe = load_fontdiffuer_pipeline(args=args)
    out_image = sampling_fs(args=args, pipe=pipe)
