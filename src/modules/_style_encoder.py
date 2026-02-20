import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from diffusers import ModelMixin
from diffusers.configuration_utils import (
    ConfigMixin,
    register_to_config,
)

# =============================================================================
#  線形代数ユーティリティ（スペクトル正規化のための power iteration）
# =============================================================================

def proj(x, y):
    """
    x を y の方向へ射影した成分を返す（行ベクトル前提で実装されている点に注意）。

    ここでの x, y は 2D テンソル（shape: [1, dim] のような行ベクトル）を想定。
    proj(x, y) = ( (y x^T) / (y y^T) ) * y
    """
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
    """
    Gram-Schmidt 直交化：
    x から、既に得られているベクトル列 ys の張る部分空間成分を引き算して直交化する。

    Args:
        x: 直交化したいベクトル（2Dテンソル、行ベクトル想定）
        ys: 既に確保されている直交ベクトルのリスト

    Returns:
        ys と直交な成分のみを残した x
    """
    for y in ys:
        x = x - proj(x, y)
    return x


def power_iteration(W, u_, update=True, eps=1e-12):
    """
    Spectral Normalization 用の power iteration（複数特異値に対応する作り）。

    W: 重み行列（2D: [out_dim, in_dim] もしくは transpose により転置されたもの）
    u_: 近似に使う左特異ベクトルのバッファのリスト（Module bufferとして保持される）
    update: True のとき、推定した u を u_ の中身に書き戻す（training時のみ更新する設計）
    eps: 正規化の数値安定用

    Returns:
        svs: 推定された特異値（最大特異値が先頭、ただし複数計算にも対応）
        us: 推定された左特異ベクトル列
        vs: 推定された右特異ベクトル列
    """
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        with torch.no_grad():
            # v = u W  （u が左特異ベクトルなら v は右特異ベクトル方向へ）
            v = torch.matmul(u, W)
            # 既存の v たちに対して直交化してから正規化（複数特異値を安定に推定するため）
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            vs += [v]

            # u = v W^T （右→左へ戻す）
            u = torch.matmul(v, W.t())
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            us += [u]

            # training 時のみバッファを更新（eval時は固定）
            if update:
                u_[i][:] = u

        # 特異値の推定： sv ≈ v W^T u^T （スカラー）
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]

    return svs, us, vs


# =============================================================================
#  MLP（線形ブロックの積み重ね）
# =============================================================================

class LinearBlock(nn.Module):
    """
    Linear + (Normalization) + (Activation) の基本ブロック。
    use_sn=True の場合は nn.utils.spectral_norm による SN を適用する。
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        norm='none',
        act='relu',
        use_sn=False
    ):
        super(LinearBlock, self).__init__()

        # Linear 層
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # PyTorch 標準の spectral_norm（こちらは重みの最大特異値を正規化）
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

        # Normalization の種類（1Dベクトル用）
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            raise AssertionError(f"Unsupported normalization: {norm}")

        # Activation の種類
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            raise AssertionError(f"Unsupported activation: {act}")

    def forward(self, x):
        """
        Args:
            x: shape [B, in_dim]
        Returns:
            out: shape [B, out_dim]
        """
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class MLP(nn.Module):
    """
    複数の LinearBlock を積んだ MLP。
    最終層は norm/act なし（回帰出力や埋め込みなどに使いやすい）。
    """
    def __init__(
        self,
        nf_in,
        nf_out,
        nf_mlp,
        num_blocks,
        norm,
        act,
        use_sn=False
    ):
        super(MLP, self).__init__()

        # ModuleList に積んで最後に Sequential 化
        self.model = nn.ModuleList()

        # 1層目：入力 → 隠れ次元
        nf = nf_mlp
        self.model.append(
            LinearBlock(nf_in, nf, norm=norm, act=act, use_sn=use_sn)
        )

        # 中間層：隠れ次元 → 隠れ次元
        for _ in range((num_blocks - 2)):
            self.model.append(
                LinearBlock(nf, nf, norm=norm, act=act, use_sn=use_sn)
            )

        # 最終層：隠れ次元 → 出力（norm/act なし）
        self.model.append(
            LinearBlock(nf, nf_out, norm='none', act='none', use_sn=use_sn)
        )

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        """
        Args:
            x: 任意 shape のテンソル。最終的に [B, -1] にフラット化して投入。
        """
        return self.model(x.view(x.size(0), -1))


# =============================================================================
#  自前実装の Spectral Normalization（Conv/Linear 用）
# =============================================================================

class SN(object):
    """
    自前の Spectral Normalization ミックスイン。

    - register_buffer を使うので、本来は nn.Module を継承したクラスと混ぜて使う前提。
    - num_svs: 推定する特異値の本数（通常 1 で最大特異値のみ）
    - num_itrs: power iteration の反復回数（1〜数回が多い）
    - transpose: W の転置を使うか（畳み込みの reshape の仕方で必要になる場合がある）
    """
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        self.num_itrs = num_itrs
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps

        # u と sv はバッファとして保持（学習パラメータではないが state_dict に含めたい）
        for i in range(self.num_svs):
            self.register_buffer(f'u{i}', torch.randn(1, num_outputs))
            self.register_buffer(f'sv{i}', torch.ones(1))

    @property
    def u(self):
        """u バッファのリストを返す。"""
        return [getattr(self, f'u{i}') for i in range(self.num_svs)]

    @property
    def sv(self):
        """sv バッファのリストを返す（ログ用などに利用できる）。"""
        return [getattr(self, f'sv{i}') for i in range(self.num_svs)]

    def W_(self):
        """
        正規化後の重みを返す。

        - self.weight を 2D 行列に reshape
        - power iteration で最大特異値 sv を近似
        - weight / sv により Lipschitz 制約（||W||_2 = 1）を近似的に強制
        """
        # Conv などの weight を [out, -1] に潰して行列として扱う
        W_mat = self.weight.view(self.weight.size(0), -1)

        # 必要なら転置（設計により out/in の扱いが逆な場合）
        if self.transpose:
            W_mat = W_mat.t()

        # power iteration を num_itrs 回
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(
                W_mat, self.u, update=self.training, eps=self.eps
            )

        # training 時のみ推定特異値をバッファへ保存（モニタ用途）
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv

        # 最大特異値 svs[0] で割って正規化した重みを返す
        return self.weight / svs[0]


class SNConv2d(nn.Conv2d, SN):
    """
    Conv2d に自前 Spectral Normalization を適用した層。
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True,
        num_svs=1, num_itrs=1, eps=1e-12
    ):
        nn.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias
        )
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        """
        正規化済み重み W_() を使って conv2d。
        """
        return F.conv2d(
            x, self.W_(), self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )

    def forward_wo_sn(self, x):
        """
        スペクトル正規化なしで畳み込み（デバッグ用/比較用）。
        """
        return F.conv2d(
            x, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )


class SNLinear(nn.Linear, SN):
    """
    Linear に自前 Spectral Normalization を適用した層。
    """
    def __init__(
        self, in_features, out_features, bias=True,
        num_svs=1, num_itrs=1, eps=1e-12
    ):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        """
        正規化済み重み W_() を使って linear。
        """
        return F.linear(x, self.W_(), self.bias)


# =============================================================================
#  ResNet 風ブロック（Discriminator/Encoder 側: DBlock, Generator 側: GBlock）
# =============================================================================

class DBlock(nn.Module):
    """
    Discriminator/Encoder 側の Residual Block（ダウンサンプル対応）。

    - which_conv を SNConv2d にするとスペクトル正規化付き conv
    - preactivation=True で Pre-activation ResNet 風の順序になる
    - downsample に AvgPool2d(2) 等を渡すことで解像度を半分にする
    """
    def __init__(
        self, in_channels, out_channels,
        which_conv=SNConv2d, wide=True,
        preactivation=False, activation=None,
        downsample=None,
    ):
        super(DBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels

        # wide=True の場合：中間チャンネル=out_channels（一般的に表現力を上げやすい）
        self.hidden_channels = self.out_channels if wide else self.in_channels

        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # メインパスの conv（ここでは kernel_size/padding は which_conv 側の partial で固定される想定）
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)

        # skip 側（ショートカット）の conv が必要かどうか
        # - チャンネル数が変わる
        # - もしくは downsample が入る（空間サイズが変わる）
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def shortcut(self, x):
        """
        ショートカット分岐（preactivation の有無で順序が変わる）。
        """
        if self.preactivation:
            # preactivation の場合は conv_sc → downsample の順にする設計
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            # 通常は downsample → conv_sc の順
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: residual 加算後の出力
        """
        # preactivation の場合、先に ReLU をかける
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x

        # メインパス
        h = self.conv1(h)
        h = self.conv2(self.activation(h))

        # ダウンサンプル（空間解像度を下げる）
        if self.downsample:
            h = self.downsample(h)

        # residual add
        return h + self.shortcut(x)


class GBlock(nn.Module):
    """
    Generator 側の Residual Block（アップサンプル対応、BatchNorm あり）。

    - which_bn を差し替えることで Conditional BN 等にも拡張可能
    - upsample に Upsample(scale_factor=2) 等を渡すと解像度を上げる
    """
    def __init__(
        self, in_channels, out_channels,
        which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d,
        activation=None, upsample=None
    ):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample

        # メインパス conv
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)

        # skip 側が必要か：チャンネル変更 or upsample
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(
                in_channels, out_channels, kernel_size=1, padding=0
            )

        # BatchNorm
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: residual 加算後の出力
        """
        # pre-activation 風：BN→Act→(Upsample)→Conv
        h = self.activation(self.bn1(x))

        # upsample があるならメイン/skip ともに空間サイズを揃えるため両方を upsample
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)

        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)

        # skip 側のチャンネル合わせ
        if self.learnable_sc:
            x = self.conv_sc(x)

        return h + x


class GBlock2(nn.Module):
    """
    Generator 側の簡易 Residual Block（BatchNorm なし、skip_connection を切れる）。

    - skip_connection=False にすると純粋な feed-forward ブロックとして使える
    - upsample 対応（GBlock と同様に両分岐を upsample）
    """
    def __init__(
        self, in_channels, out_channels,
        which_conv=nn.Conv2d, activation=None,
        upsample=None, skip_connection=True
    ):
        super(GBlock2, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv = which_conv
        self.activation = activation
        self.upsample = upsample

        # メインパス conv
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)

        # skip 側が必要か：チャンネル変更 or upsample
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(
                in_channels, out_channels, kernel_size=1, padding=0
            )

        self.skip_connection = skip_connection

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: skip_connection に応じて residual add する/しない
        """
        h = self.activation(x)

        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)

        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)

        if self.learnable_sc:
            x = self.conv_sc(x)

        # residual を足すかどうかを切り替え可能
        if self.skip_connection:
            out = h + x
        else:
            out = h
        return out


# =============================================================================
#  StyleEncoder のアーキテクチャ定義（入力解像度ごとのチャンネル設計）
# =============================================================================

def style_encoder_textedit_addskip_arch(ch=64, out_channel_multiplier=1, input_nc=3):
    """
    解像度ごとの Encoder の in/out チャンネル構成を返す。

    戻り値 arch[resolution] は以下の辞書：
      - in_channels: 各ブロックの入力チャンネル列
      - out_channels: 各ブロックの出力チャンネル列
      - resolution: 各段の解像度（ダウンサンプル後の H=W）
    """
    arch = {}

    # 96x96 入力：96→48→24→12→6→3 という 5段ダウンサンプル想定
    arch[96] = {
        'in_channels':   [input_nc] + [ch * item for item in [1, 2, 4, 8]],
        'out_channels':  [item * ch for item in [1, 2, 4, 8, 16]],
        'resolution':    [48, 24, 12, 6, 3],
    }

    # 128x128 入力：128→64→32→16→8→4
    arch[128] = {
        'in_channels':   [input_nc] + [ch * item for item in [1, 2, 4, 8]],
        'out_channels':  [item * ch for item in [1, 2, 4, 8, 16]],
        'resolution':    [64, 32, 16, 8, 4],
    }

    # 256x256 入力：256→128→64→32→16→8→4（ブロック数が多い）
    arch[256] = {
        'in_channels':   [input_nc] + [ch * item for item in [1, 2, 4, 8, 8]],
        'out_channels':  [item * ch for item in [1, 2, 4, 8, 8, 16]],
        'resolution':    [128, 64, 32, 16, 8, 4],
    }

    return arch


# =============================================================================
#  StyleEncoder（Diffusers の ModelMixin + ConfigMixin）
# =============================================================================

class StyleEncoder(ModelMixin, ConfigMixin):
    """
    スタイル画像（例：フォントの参照画像）を埋め込みに変換する Encoder。

    仕様（コメントの例）:
        Input : [B, 3, 128, 128]
        Output:
          - style_emd:  途中の特徴マップ（最終段）例：[B, C, 4, 4]
          - h:          グローバル特徴（pool後のベクトル）例：[B, C]
          - residual_features: 途中段の特徴マップのリスト（skip 用など）
    """
    @register_to_config
    def __init__(
        self,
        G_ch=64,
        G_wide=True,
        resolution=128,
        G_kernel_size=3,
        G_attn='64_32_16_8',   # このコード断片では attention 自体は未使用（設定だけ保持）
        n_classes=1000,        # 同上（拡張用の設定が残っている可能性）
        num_G_SVs=1,
        num_G_SV_itrs=1,
        G_activation=nn.ReLU(inplace=False),
        SN_eps=1e-12,
        output_dim=1,
        G_fp16=False,
        G_init='N02',
        G_param='SN',
        nf_mlp=512,
        nEmbedding=256,
        input_nc=3,
        output_nc=3
    ):
        super(StyleEncoder, self).__init__()

        # -------------------------
        # 設定値（ConfigMixin で保存される）
        # -------------------------
        self.ch = G_ch
        self.G_wide = G_wide
        self.resolution = resolution
        self.kernel_size = G_kernel_size
        self.attention = G_attn
        self.n_classes = n_classes
        self.activation = G_activation
        self.init = G_init
        self.G_param = G_param
        self.SN_eps = SN_eps
        self.fp16 = G_fp16

        # -------------------------
        # 途中特徴を保存する段のインデックス
        # （実装上、DBlock を積んだ後の各段を residual_features に入れる）
        # -------------------------
        if self.resolution == 96:
            self.save_featrues = [0, 1, 2, 3, 4]
        if self.resolution == 128:
            self.save_featrues = [0, 1, 2, 3, 4]
        elif self.resolution == 256:
            self.save_featrues = [0, 1, 2, 3, 4, 5]

        # ※ out_channel_multiplier のスペルが out_channel_nultipiler になっているが、
        #   実際には 1 固定で使っている（将来的な拡張用の名残と思われる）
        self.out_channel_nultipiler = 1

        # 解像度ごとのアーキテクチャ定義を取得
        self.arch = style_encoder_textedit_addskip_arch(
            self.ch,
            self.out_channel_nultipiler,
            input_nc
        )[resolution]

        # -------------------------
        # Conv/Linear の実装（SN あり/なし）
        # -------------------------
        if self.G_param == 'SN':
            # SNConv2d / SNLinear を partial で固定引数つきにして扱う
            self.which_conv = functools.partial(
                SNConv2d,
                kernel_size=3, padding=1,   # 3x3 conv + same padding
                num_svs=num_G_SVs,
                num_itrs=num_G_SV_itrs,
                eps=self.SN_eps
            )
            self.which_linear = functools.partial(
                SNLinear,
                num_svs=num_G_SVs,
                num_itrs=num_G_SV_itrs,
                eps=self.SN_eps
            )
        else:
            # ここに通常 Conv2d/Linear を置く拡張余地がある（現状未実装）
            raise NotImplementedError("G_param != 'SN' is not implemented in this snippet.")

        # -------------------------
        # DBlock を段階的に積む（各段で AvgPool2d(2) によりダウンサンプル）
        # -------------------------
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[
                DBlock(
                    in_channels=self.arch['in_channels'][index],
                    out_channels=self.arch['out_channels'][index],
                    which_conv=self.which_conv,
                    wide=self.G_wide,
                    activation=self.activation,
                    preactivation=(index > 0),       # 1段目だけ preactivation=False
                    downsample=nn.AvgPool2d(2)       # 毎段 H,W を 1/2 にする
                )
            ]]

        # blocks は「段」→「段内のブロックリスト」という二重構造（将来、複数ブロック/段に対応しやすい）
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # -------------------------
        # 最終段の追加処理：
        # InstanceNorm2d → Activation → 1x1 Conv
        # （DBlock の後の特徴を整形し、style_emd を得る）
        # -------------------------
        last_layer = nn.Sequential(
            nn.InstanceNorm2d(self.arch['out_channels'][-1]),
            self.activation,
            nn.Conv2d(
                self.arch['out_channels'][-1],
                self.arch['out_channels'][-1],
                kernel_size=1,
                stride=1
            )
        )

        # self.blocks の最後に追加（forward では self.blocks[-1] として別扱いされる）
        self.blocks.append(last_layer)

        # 重み初期化
        self.init_weights()

    def init_weights(self):
        """
        畳み込み/線形/Embedding の重み初期化をまとめて実行する。
        self.init に応じて初期化法を切り替える。
        """
        self.param_count = 0
        for module in self.modules():
            if (
                isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)
            ):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')

                # パラメータ数をカウント（bias も含む）
                self.param_count += sum([p.data.nelement() for p in module.parameters()])

        # ここは元コードの表記を維持（D''s は discriminator の名残っぽい）
        print("Param count for D's initialized parameters: %d" % self.param_count)

    def forward(self, x):
        """
        Args:
            x: スタイル画像 [B, C(=3), H, W]

        Returns:
            style_emd:  最終特徴マップ（last_layer 適用後） [B, C_last, H_last, W_last]
            h:          グローバル埋め込み（adaptive avg pool→flatten） [B, C_last]
            residual_features: 途中段の特徴（skip 用など）を格納したリスト
                - 先頭には入力 x を入れている（residual_features[0] = x）
                - 各段の出力 h を save_featrues に応じて追加
        """
        h = x

        # 途中特徴を蓄える（skip connection や multi-scale 参照に使う想定）
        residual_features = []
        residual_features.append(h)  # 入力そのものも保存

        # self.blocks のうち、最後の last_layer 以外を順に適用
        for index, blocklist in enumerate(self.blocks):
            # last_layer は nn.Sequential なので、ここで blocklist がそれに当たる可能性があるが
            # 下の "if index in self.save_featrues[:-1]" で最後を除外している前提の設計になっている
            for block in blocklist:
                h = block(h)

            # 指定された段の特徴を保存（最後の段は forward 後半で style_emd として扱う）
            if index in self.save_featrues[:-1]:
                residual_features.append(h)

        # 明示的に last_layer を適用（self.blocks[-1] が last_layer）
        h = self.blocks[-1](h)

        # style_emd は空間情報を保持したまま返す（後段で cross-attn 等に使う想定が多い）
        style_emd = h

        # グローバル埋め込み：空間平均で [B, C, 1, 1] → [B, C]
        h = F.adaptive_avg_pool2d(h, (1, 1))
        h = h.view(h.size(0), -1)

        return style_emd, h, residual_features
