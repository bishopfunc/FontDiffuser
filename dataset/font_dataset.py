import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def get_nonorm_transform(resolution):
    nonorm_transform = transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    return nonorm_transform


class FontDataset(Dataset):
    """The dataset of font generation"""

    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg

        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        self.target_images = []
        # images with related style
        self.style_to_images = {}
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        for style in os.listdir(target_image_dir):
            images_related_style = []
            for img in os.listdir(f"{target_image_dir}/{style}"):
                img_path = f"{target_image_dir}/{style}/{img}"
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            self.style_to_images[style] = images_related_style

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split("/")[-1]
        style, content = target_image_name.split(".")[0].split("+")
        # content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.jpg"
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.png"

        # Skip if pair of content image and target image does not exist
        if not os.path.exists(content_image_path) and os.path.exists(target_image_path):
            print(
                f"Warning: Content image not found for target image '{target_image_path}': {content_image_path}"
            )
            return self.__getitem__((index + 1) % len(self.target_images))

        # Read content image
        content_image = Image.open(content_image_path).convert("RGB")

        # Random sample used for style image
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)
        style_image_path = random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")

        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)

        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
        }

        if self.scr:
            # styles available
            style_list = list(self.style_to_images.keys())

            # 自分のstyleを除外
            if style in style_list:
                style_list.remove(style)
            else:
                # ここは状況的にあり得るのでログ出しておくとデバッグしやすい
                print(
                    f"[WARN] style '{style}' not in style_list (len={len(style_list)})",
                    flush=True,
                )

            # 取れる負例数にクリップ
            k = min(self.num_neg, len(style_list))
            if k == 0:
                # 負例を作れない → SCRをスキップするか、例外にするか選ぶ
                # まずは学習続行したいなら「負例なし」で返すのが無難
                choose_neg_names = []
            else:
                neg_styles = random.sample(style_list, k)  # 重複なし
                choose_neg_names = [
                    f"{self.root}/train/TargetImage/{s}/{s}+{content}.png"
                    for s in neg_styles
                ]
                # choose_neg_names = [
                #     f"{self.root}/train/TargetImage/{s}/{s}+{content}.jpg"
                #     for s in neg_styles
                # ]

            # Load neg_images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat(
                        [neg_images, neg_image[None, :, :, :]], dim=0
                    )
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)
