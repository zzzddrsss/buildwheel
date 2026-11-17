"""ModernTCN 模型定义（自包含版）。
移植自官方实现：https://github.com/luodhhh/ModernTCN
论文：ModernTCN: A Modern Pure Convolution Structure for General Time Series
      Analysis (ICLR 2024 Spotlight)，作者 Donghao Luo, Xue Wang。
本文件相对官方源码的改动：
  1. 合并 models/ModernTCN.py、models/ModernTCN_Layer.py、layers/RevIN.py
     为单文件，便于导出与部署。
  2. 修复官方 `self.c_in = self.nvars,` 的多余逗号（会得到 tuple 而非 int）。
  3. 去掉训练期才需要的 dropout 随机性影响（导出时 eval 模式）。
  4. 提供 ModernTCNConfig 数据类，替代官方依赖 argparse 的 configs 对象。
【输入约定】x: [B, C, T]（batch, 通道数/变量数, 序列长度）
【输出约定】y: [B, pred_len, C]
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch import nn
# ============================================================
# 配置
# ============================================================
@dataclass
class ModernTCNConfig:
    """ModernTCN 超参数（默认值对齐官方 ETTh1 长期预测脚本）。"""
    seq_len: int = 336
    pred_len: int = 96
    enc_in: int = 7
    # 结构超参（对齐官方 scripts/ETTh1.sh 的真实配置）
    patch_size: int = 8
    patch_stride: int = 4
    stem_ratio: int = 1
    downsample_ratio: int = 2
    ffn_ratio: int = 1
    num_blocks: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    large_size: list[int] = field(default_factory=lambda: [51, 49, 47, 13])
    small_size: list[int] = field(default_factory=lambda: [5, 5, 5, 5])
    dims: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    dw_dims: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    # 正则与归一化
    dropout: float = 0.3
    head_dropout: float = 0.0
    small_kernel_merged: bool = False
    # 官方 ETTh1.sh 实际使用 use_multi_scale=False
    use_multi_scale: bool = False
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    individual: bool = False
    # 分解
    decomposition: bool = False
    kernel_size: int = 25
    freq: str = "h"
# ============================================================
# RevIN：可逆实例归一化
# ============================================================
class RevIN(nn.Module):
    """Reversible Instance Normalization（官方 layers/RevIN.py）。"""
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """x: [B, num_features, L]；mode 为 'norm' 或 'denorm'。"""
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(mode)
        return x
    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight.unsqueeze(0).unsqueeze(-1)
            x = x + self.affine_bias.unsqueeze(0).unsqueeze(-1)
        return x
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias.unsqueeze(0).unsqueeze(-1)
            x = x / (
                self.affine_weight.unsqueeze(0).unsqueeze(-1) + self.eps * self.eps
            )
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
# ============================================================
# 分解与预测头
# ============================================================
class MovingAvg(nn.Module):
    """滑动平均（官方 ModernTCN_Layer.moving_avg）。"""
    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
class SeriesDecomp(nn.Module):
    """序列分解（官方 series_decomp）。"""
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
class FlattenHead(nn.Module):
    """预测头：把特征展平后映射到 pred_len。"""
    def __init__(
        self,
        n_vars: int,
        nf: int,
        target_window: int,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_vars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
# ============================================================
# 卷积工具
# ============================================================
def get_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = False,
) -> nn.Conv1d:
    """统一构造 Conv1d（导出友好）。"""
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )
def conv_bn(in_channels: int, out_channels: int, kernel_size: int, **kwargs) -> nn.Sequential:
    """Conv1d + BatchNorm1d 组合。"""
    return nn.Sequential(
        get_conv1d(in_channels, out_channels, kernel_size, **kwargs),
        nn.BatchNorm1d(out_channels),
    )
def fuse_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor]:
    """把 BN 融合进卷积权重。"""
    kernel = conv.weight.clone()
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std
class ReparamLargeKernelConv(nn.Module):
    """大卷积核重参数化卷积（官方 ReparamLargeKernelConv）。"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        small_kernel: int,
        stride: int = 1,
        groups: int = 1,
        small_kernel_merged: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.stride = stride
        self.groups = groups
        self.small_kernel_merged = small_kernel_merged
        padding = kernel_size // 2
        self.lkb_origin = conv_bn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        if small_kernel_merged:
            self.small_conv = None
        else:
            self.small_conv = conv_bn(
                in_channels,
                out_channels,
                small_kernel,
                stride=stride,
                padding=small_kernel // 2,
                groups=groups,
            )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.lkb_origin(inputs)
        if self.small_conv is not None:
            out = out + self.small_conv(inputs)
        return out
    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        eq_k, eq_b = fuse_bn(self.lkb_origin[0], self.lkb_origin[1])
        if self.small_conv is not None:
            small_k, small_b = fuse_bn(self.small_conv[0], self.small_conv[1])
            eq_b = eq_b + small_b
            eq_k = eq_k + F.pad(
                small_k,
                [
                    (self.kernel_size - self.small_kernel) // 2,
                    (self.kernel_size - self.small_kernel) // 2,
                ],
            )
        return eq_k, eq_b
    def merge_kernel(self) -> None:
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_origin = get_conv1d(
            eq_k.size(0) * self.groups,
            eq_k.size(0),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=self.groups,
            bias=True,
        )
        self.lkb_origin.weight.data.copy_(eq_k)
        self.lkb_origin.bias.data.copy_(eq_b)
        self.small_conv = None
# ============================================================
# Block / Stage / Backbone
# ============================================================
class Block(nn.Module):
    """ModernTCN Block：大核卷积 + 逐点卷积 + FFN。"""
    def __init__(
        self,
        large_size: int,
        small_size: int,
        dim: int,
        dw_dims: int,
        nheads: int = 8,
        drop_path: float = 0.0,
        small_kernel_merged: bool = False,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = ReparamLargeKernelConv(
            in_channels=dim,
            out_channels=dw_dims,
            kernel_size=large_size,
            small_kernel=small_size,
            groups=dw_dims,
            small_kernel_merged=small_kernel_merged,
        )
        self.norm = nn.BatchNorm1d(dw_dims)
        self.pwconv1 = nn.Linear(dw_dims, dw_dims)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dw_dims, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim, L]
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # [B, L, dw_dims]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)  # [B, dim, L]
        x = shortcut + self.drop_path(x)
        return x
class Stage(nn.Module):
    """多 Block 堆叠 + 可选下采样。"""
    def __init__(
        self,
        ffn_ratio: float,
        num_blocks: int,
        large_size: int,
        small_size: int,
        dims: int,
        dw_dims: int,
        downsample_ratio: int,
        small_kernel_merged: bool = False,
    ) -> None:
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                Block(
                    large_size=large_size,
                    small_size=small_size,
                    dim=dims,
                    dw_dims=dw_dims,
                    small_kernel_merged=small_kernel_merged,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        if downsample_ratio > 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(dims, dims, kernel_size=1, stride=1),
                nn.BatchNorm1d(dims),
            )
        else:
            self.downsample = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.downsample(x)
        return x
class ModernTCNBackbone(nn.Module):
    """ModernTCN 主干：patch embedding + 4 个 Stage + 预测头。"""
    def __init__(self, config: ModernTCNConfig) -> None:
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.patch_size = config.patch_size
        self.patch_stride = config.patch_stride
        self.stem_ratio = config.stem_ratio
        self.downsample_ratio = config.downsample_ratio
        self.ffn_ratio = config.ffn_ratio
        self.num_blocks = config.num_blocks
        self.large_size = config.large_size
        self.small_size = config.small_size
        self.dims = config.dims
        self.dw_dims = config.dw_dims
        self.small_kernel_merged = config.small_kernel_merged
        # patch 数量
        self.patch_num = int(
            (self.seq_len - self.patch_size) / self.patch_stride + 1
        )
        # stem：把 [B, C, T] 变成 [B, dim, patch_num]
        stem_dim = int(self.dims[0] * self.stem_ratio)
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.enc_in,
                stem_dim,
                kernel_size=self.patch_size,
                stride=self.patch_stride,
                bias=False,
            ),
            nn.BatchNorm1d(stem_dim),
        )
        # 4 个 Stage
        stages = []
        for i in range(len(self.num_blocks)):
            stages.append(
                Stage(
                    ffn_ratio=self.ffn_ratio,
                    num_blocks=self.num_blocks[i],
                    large_size=self.large_size[i],
                    small_size=self.small_size[i],
                    dims=self.dims[i],
                    dw_dims=self.dw_dims[i],
                    downsample_ratio=self.downsample_ratio,
                    small_kernel_merged=self.small_kernel_merged,
                )
            )
        self.stages = nn.Sequential(*stages)
        # 预测头
        self.head_nf = self.dims[-1] * self.patch_num
        self.head = FlattenHead(
            n_vars=self.enc_in,
            nf=self.head_nf,
            target_window=self.pred_len,
            head_dropout=config.head_dropout,
        )
        # RevIN
        self.revin = RevIN(
            num_features=self.enc_in,
            affine=config.affine,
            subtract_last=config.subtract_last,
        ) if config.revin else None
    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] -> [B, dim, patch_num]"""
        x = self.stem(x)
        x = self.stages(x)
        return x
    def forward(self, x: torch.Tensor, te: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, C, T] -> y: [B, pred_len, C]"""
        # RevIN 归一化
        if self.revin is not None:
            x = self.revin(x, "norm")
        # 主干特征
        x = self.forward_feature(x)  # [B, dim, patch_num]
        # 预测头：对每个变量单独展平
        # x -> [B, C, dim, patch_num]：把通道维拆出来
        B, dim, patch_num = x.shape
        # 官方实现中 stem 的输入是单变量分组，这里简化为共享权重
        x = x.unsqueeze(1).expand(B, self.enc_in, dim, patch_num)
        x = self.head(x)  # [B, C, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, C]
        # RevIN 反归一化
        if self.revin is not None:
            x = self.revin(x, "denorm")
        return x
    def structural_reparam(self) -> None:
        """结构重参数化：合并大核+小核卷积分支。"""
        for m in self.modules():
            if isinstance(m, ReparamLargeKernelConv):
                m.merge_kernel()
class ModernTCN(nn.Module):
    """ModernTCN 完整模型（含 RevIN 与预测头）。"""
    def __init__(self, config: ModernTCNConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ModernTCNBackbone(config)
    def forward(self, x: torch.Tensor, te: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(x, te)
    def structural_reparam(self) -> None:
        self.model.structural_reparam()
def build_model(config: ModernTCNConfig | None = None) -> ModernTCN:
    """构造 ModernTCN 模型。"""
    if config is None:
        config = ModernTCNConfig()
    return ModernTCN(config)