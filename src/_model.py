from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """
    FontDiffuser 用の DPM（Diffusion Probabilistic Model）forward をまとめたラッパーモデル。

    役割：
      - cond から content 参照画像と style 参照画像を受け取る
      - style_encoder / content_encoder で特徴抽出する
      - UNet に「複数種類の条件」を encoder_hidden_states として渡す
      - UNet の出力（通常は predicted noise）を返す

    想定している条件情報（input_hidden_states）：
      1) style_img_feature:
         style encoder の最終特徴マップ（空間情報を持つ2D特徴）
      2) content_residual_features:
         content encoder のマルチスケール特徴（skip 用）
      3) style_hidden_states:
         style_img_feature を (B, HW, C) へ変形したトークン列（cross-attn 用など）
      4) style_content_res_features:
         「style画像を content encoder に通した」マルチスケール特徴
         （style画像が持つ“文字形状/内容”側の情報を抽出して条件付けに使う意図）
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
    ):
        """
        Args:
            unet:
                Diffusers 互換の UNet モデル。
                forward(x_t, timesteps, encoder_hidden_states=..., ...) を受け取る想定。
            style_encoder:
                スタイル参照画像を特徴に変換するエンコーダ（StyleEncoder）。
            content_encoder:
                内容（文字形状）参照画像を特徴に変換するエンコーダ。
        """
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
        """
        Args:
            x_t:
                拡散過程の時刻 t におけるノイズ付き画像（もしくは潜在）テンソル。
                典型的には shape: [B, C, H, W]
            timesteps:
                現在の拡散ステップ（scalar または [B]）。
                Diffusers の UNet が受け取る形式に合わせる。
            cond:
                条件入力のタプル/リスト。
                cond[0] = content_images（内容参照）
                cond[1] = style_images（スタイル参照）
            content_encoder_downsample_size:
                content encoder 側の downsample 解像度等を UNet に伝えるための追加引数。
                （UNet 内でスキップ特徴の解像度合わせ等に使う想定）
            version:
                版管理用の引数。※この forward では未使用（将来分岐用の名残の可能性）。

        Returns:
            noise_pred:
                UNet が予測したノイズ（epsilon）など。shape は通常 x_t と同じ。
        """
        # ------------------------------------------------------------
        # 1) 条件画像を取り出す
        # ------------------------------------------------------------
        content_images = cond[0]  # 内容（文字形状）参照画像
        style_images = cond[1]  # スタイル参照画像（フォントの雰囲気など）

        # ------------------------------------------------------------
        # 2) style encoder で style 画像を特徴抽出
        #    - style_img_feature: 最終特徴マップ（空間情報あり）
        #    - style_residual_features: マルチスケール特徴（skip 用など）
        # ------------------------------------------------------------
        style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        print("style_images.shape:", style_images.shape)
        print("style_img_feature.shape:", style_img_feature.shape)

        # style_img_feature: [B, C, H, W]
        batch_size, channel, height, width = style_img_feature.shape

        # ------------------------------------------------------------
        # 3) style の特徴マップを token 列 (B, HW, C) に変形
        #    - cross attention の encoder_hidden_states として使いやすい表現
        # ------------------------------------------------------------
        # permute: [B, C, H, W] -> [B, H, W, C]
        # reshape: [B, H, W, C] -> [B, H*W, C]
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # ------------------------------------------------------------
        # 4) content encoder で content 参照画像を特徴抽出
        #    - content_img_feture: 最終特徴（空間特徴マップ等）
        #    - content_residual_features: マルチスケール特徴
        # ------------------------------------------------------------
        content_img_feture, content_residual_features = self.content_encoder(
            content_images
        )

        # content_residual_features に最終特徴も追加しておく
        # （UNet 側で「段ごとの特徴＋最終特徴」を一括で扱う設計が多い）
        content_residual_features.append(content_img_feture)

        # ------------------------------------------------------------
        # 5) style 画像を content encoder にも通す（= style 画像の“内容/形状”特徴を抽出）
        #    - 例：スタイル参照画像にも「どの文字が描かれているか」が含まれるため、
        #      その形状情報を別経路で条件付けに使う意図がある
        # ------------------------------------------------------------
        style_content_feature, style_content_res_features = self.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        # ------------------------------------------------------------
        # 6) UNet に渡す条件をパッキング
        #    encoder_hidden_states に「複数種類の条件」をまとめて渡している点が特徴。
        #    ※通常の Diffusers では Tensor を渡すことが多いが、ここは UNet 側が
        #      list を解釈できるよう独自拡張されている前提。
        # ------------------------------------------------------------
        input_hidden_states = [
            style_img_feature,  # (1) 空間特徴マップとしての style
            content_residual_features,  # (2) content の multi-scale / skip 用特徴
            style_hidden_states,  # (3) token 列としての style（cross-attn 等）
            style_content_res_features,  # (4) style を content encoder に通した特徴
        ]

        # ------------------------------------------------------------
        # 7) UNet forward
        #    - x_t と timesteps に加え、条件情報 encoder_hidden_states を渡す
        # ------------------------------------------------------------
        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )

        # Diffusers の UNet は通常 out.sample を返すが、
        # この実装では out が tuple/list で先頭がノイズ予測になっている想定。
        noise_pred = out[0]

        # ------------------------------------------------------------
        # 8) predicted noise を返す（DPM の損失で使う）
        # ------------------------------------------------------------
        return noise_pred
