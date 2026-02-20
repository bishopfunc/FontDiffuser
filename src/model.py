import torch
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    def forward(
        self,
        x_t,
        timesteps,
        style_images,
        content_images,
        content_encoder_downsample_size,
    ):
        # Get the style feature
        style_img_feature, _, _ = self.style_encoder(style_images)
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # Get the content feature
        content_img_feature, content_residual_features = self.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feature)

        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
        ]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]

        return noise_pred, offset_out_sum


class FontDiffuserModelDPMFewShot(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    def forward(
        self,
        x_t,
        timesteps,
        cond,
        content_encoder_downsample_size,
        version,
    ):
        content_images = cond[0]  # torch.Size([2, 3, 96, 96]),
        style_images_stacked = cond[1]  # torch.Size([1 + N, 3, 96, 96])
        _first_style_images = style_images_stacked[0].unsqueeze(
            0
        )  # torch.Size([1, 3, 96, 96])
        _other_style_images_stacked = style_images_stacked[
            1:
        ]  # torch.Size([N, 3, 96, 96])

        # prepare style_images for style encoder
        style_images_list = []
        for _other_style_images in _other_style_images_stacked:
            _other_style_images = _other_style_images.unsqueeze(
                0
            )  # torch.Size([1, 3, 96, 96])
            _style_images = torch.cat(
                [_first_style_images, _other_style_images], dim=0
            )  # torch.Size([2, 3, 96, 96])
            style_images_list.append(_style_images)

        # Get style feature
        style_img_feature_list = []
        for style_images in style_images_list:
            style_img_feature, _, _ = self.style_encoder(
                style_images
            )  # torch.Size([2, 1024, 3, 3])
            style_img_feature_list.append(style_img_feature)

        style_img_feature = torch.mean(
            torch.stack(style_img_feature_list, dim=0), dim=0
        )  # torch.Size([2, 1024, 3, 3])

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )  # torch.Size([2, 9, 1024])

        # Get content feature
        content_img_feature, content_residual_features = self.content_encoder(
            content_images
        )
        content_residual_features.append(
            content_img_feature
        )  # torch.Size([2, 256, 12, 12])

        # Get the content feature from reference image
        mode = "mean"
        if mode == "first":
            style_content_feature, style_content_res_features = self.content_encoder(
                style_images_list[0]
            )
            style_content_res_features.append(
                style_content_feature  # torch.Size([2, 256, 12, 12])
            )
        elif mode == "mean":
            style_content_feature_list = []
            for style_images in style_images_list:
                style_content_feature, style_content_res_features = (
                    self.content_encoder(style_images)
                )  # torch.Size([2, 256, 12, 12])
                style_content_feature_list.append(style_content_feature)
            style_content_feature = torch.mean(
                torch.stack(style_content_feature_list, dim=0), dim=0
            )  # torch.Size([2, 256, 12, 12])
            style_content_res_features.append(style_content_feature)
        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
        ]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]

        return noise_pred


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    def forward(
        self,
        x_t,
        timesteps,
        cond,
        content_encoder_downsample_size,
        version,
    ):
        content_images = cond[0]  # torch.Size([2, 3, 96, 96]),
        style_images = cond[1]  # torch.Size([2, 3, 96, 96]),

        # Get style feature
        style_img_feature, _, _ = self.style_encoder(
            style_images
        )  # torch.Size([2, 1024, 3, 3])

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )  # torch.Size([2, 9, 1024])

        # Get content feature
        content_img_feature, content_residual_features = self.content_encoder(
            content_images
        )
        content_residual_features.append(
            content_img_feature
        )  # torch.Size([2, 256, 12, 12])

        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(
            style_images
        )
        style_content_res_features.append(
            style_content_feature
        )  # torch.Size([2, 256, 12, 12])

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
        ]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]

        return noise_pred
