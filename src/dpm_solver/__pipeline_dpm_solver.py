
import torch
from PIL import Image

from .dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP, model_wrapper


class FontDiffuserDPMPipeline:
    """FontDiffuser pipeline with DPM_Solver scheduler."""

    def __init__(
        self,
        model,
        ddpm_train_scheduler,
        version="V3",
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=7.5,
    ):
        super().__init__()
        self.model = model
        self.train_scheduler_betas = ddpm_train_scheduler.betas
        # Define the noise schedule
        self.noise_schedule = NoiseScheduleVP(
            schedule="discrete", betas=self.train_scheduler_betas
        )

        self.version = version
        self.model_type = model_type
        self.guidance_type = guidance_type
        self.guidance_scale = guidance_scale

    def numpy_to_pil(self, images):
        """Convert a numpy image or a batch of images to a PIL image."""
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def generate(
        self,
        content_images,
        style_images,
        batch_size,
        order,
        num_inference_step,
        content_encoder_downsample_size,
        t_start=None,
        t_end=None,
        dm_size=(96, 96),
        algorithm_type="dpmsolver++",
        skip_type="time_uniform",
        method="multistep",
        correcting_x0_fn=None,
        generator=None,
        merge_style_refs="mean",  # ★追加: "mean" / None
    ):
        model_kwargs = {}
        model_kwargs["version"] = self.version
        model_kwargs["content_encoder_downsample_size"] = (
            content_encoder_downsample_size
        )

        # ------------------------------------------------------------
        # ★ few-shot style: [B, N, C, H, W] が来たら flatten してバッチ化
        # ------------------------------------------------------------
        orig_B = content_images.shape[0]
        N = 1
        if style_images.dim() == 5:
            # style_images: [B, N, C, H, W]
            B, N, C, H, W = style_images.shape
            assert B == orig_B, f"content B={orig_B}, style B={B} mismatch"

            # content を N 回繰り返して [B*N, C, H, W] にする
            content_images = content_images.repeat_interleave(N, dim=0)
            # style を [B*N, C, H, W] にする
            style_images = style_images.reshape(B * N, C, H, W)

            # batch_size も実体に合わせて上書き
            batch_size = B * N

        # cond/uncond を作る（以降は従来通り）
        cond = [content_images, style_images]

        uncond_content_images = torch.ones_like(content_images).to(self.model.device)
        uncond_style_images = torch.ones_like(style_images).to(self.model.device)
        uncond = [uncond_content_images, uncond_style_images]

        model_fn = model_wrapper(
            model=self.model,
            noise_schedule=self.noise_schedule,
            model_type=self.model_type,
            model_kwargs=model_kwargs,
            guidance_type=self.guidance_type,
            condition=cond,
            unconditional_condition=uncond,
            guidance_scale=self.guidance_scale,
        )

        dpm_solver = DPM_Solver(
            model_fn=model_fn,
            noise_schedule=self.noise_schedule,
            algorithm_type=algorithm_type,
            correcting_x0_fn=correcting_x0_fn,
        )

        x_T = torch.randn(
            (batch_size, 3, dm_size[0], dm_size[1]),
            generator=generator,
        ).to(self.model.device)

        x_sample = dpm_solver.sample(
            x=x_T,
            steps=num_inference_step,
            order=order,
            skip_type=skip_type,
            method=method,
        )

        # ------------------------------------------------------------
        # ★ N 個生成したものを 1 個にマージ（B に戻す）
        # ------------------------------------------------------------
        if N > 1 and merge_style_refs == "mean":
            # x_sample: [B*N, 3, H, W] -> [B, N, 3, H, W] -> mean -> [B, 3, H, W]
            x_sample = x_sample.reshape(orig_B, N, *x_sample.shape[1:]).mean(dim=1)
            batch_size = orig_B  # 戻す

        x_sample = (x_sample / 2 + 0.5).clamp(0, 1)
        x_sample = x_sample.cpu().permute(0, 2, 3, 1).numpy()
        x_images = self.numpy_to_pil(x_sample)
        return x_images
