"""ModernTCN 模型定义（自包含版，ExecuTorch 导出友好）。
移植自官方实现：https://github.com/luodhhh/ModernTCN
论文：ModernTCN: A Modern Pure Convolution Structure for General Time Series
      Analysis (ICLR 2024 Spotlight)，作者 Donghao Luo, Xue Wang。
【输入约定】x: [B, C, T]   (batch, 通道数/变量数, 序列长度)
【输出约定】y: [B, pred_len, C]
【本版关键修复】
  1. ReparamLargeKernelConv 的 groups 语义修正：
     - 深度卷积分支 groups = in_channels（逐通道卷积）
     - merge_kernel 不再错误地乘 groups
     - 解决 "expected input to have 4096 channels" 报错
  2. RevIN 的 mean/stdev 用局部变量传递，不挂 self 属性，
     解决 torch.export 的 "assigned during export" 警告
  3. RevIN 归一化 [B,C,T] 与反归一化 [B,pred_len,C] 的
     affine 广播维度分别处理，解决 "size of tensor a (96) must
     match tensor b (7)" 报错
"""
from __future__ import annotations
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
    patch_size: int = 8
    patch_stride: int = 4
    stem_ratio: int = 1
    downsample_ratio: int = 2
    ffn_ratio: int = 1
    num_blocks: list = field(default_factory=lambda: [1, 1, 1, 1])
    large_size: list = field(default_factory=lambda: [51, 49, 47, 13])
    small_size: list = field(default_factory=lambda: [5, 5, 5, 5])
    dims: list = field(default_factory=lambda: [64, 64, 64, 64])
    dw_dims: list = field(default_factory=lambda: [64, 64, 64, 64])
    dropout: float = 0.3
    head_dropout: float = 0.0
    small_kernel_merged: bool = False
    use_multi_scale: bool = False
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    individual: bool = False
    decomposition: bool = False
    kernel_size: int = 25
    freq: str = "h"
# ============================================================
# RevIN：可逆实例归一化
# ============================================================
class RevIN(nn.Module):
    """Reversible Instance Normalization。
    归一化输入  x: [B, C, T]        —— 通道维在 dim=1
    反归一化输入 x: [B, pred_len, C] —— 通道维在 dim=2
    """
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
    def get_statistics(self, x: torch.Tensor):
        """计算统计量，返回 (mean, stdev)，形状均为 [B, C, 1]。"""
        # x: [B, C, T]
        if self.subtract_last:
            mean = x[:, :, -1:].detach()          # [B, C, 1]
        else:
            mean = torch.mean(x, dim=-1, keepdim=True)   # [B, C, 1]
        stdev = torch.sqrt(
            torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps
        )                                                  # [B, C, 1]
        return mean, stdev
    def normalize(self, x: torch.Tensor):
        """x: [B, C, T] -> (归一化结果 [B,C,T], mean [B,C,1], stdev [B,C,1])"""
        mean, stdev = self.get_statistics(x)
        y = (x - mean) / stdev
        if self.affine:
            # affine 参数 [C] -> [1, C, 1]，对齐通道维 dim=1
            y = y * self.affine_weight.reshape(1, -1, 1)
            y = y + self.affine_bias.reshape(1, -1, 1)
        return y, mean, stdev
    def denormalize(self, x: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor):
        """x: [B, pred_len, C]；mean/stdev: [B, C, 1] -> 转 [B, 1, C] 对齐"""
        mean = mean.transpose(1, 2)      # [B, C, 1] -> [B, 1, C]
        stdev = stdev.transpose(1, 2)    # [B, C, 1] -> [B, 1, C]
        if self.affine:
            # affine 参数 [C] -> [1, 1, C]，对齐通道维 dim=2
            w = self.affine_weight.reshape(1, 1, -1)
            b = self.affine_bias.reshape(1, 1, -1)
            x = x - b
            x = x / (w + self.eps * self.eps)
        x = x * stdev
        x = x + mean
        return x
# ============================================================
# 卷积工具
# ============================================================
def fuse_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    """把 BN 融合进卷积权重，返回 (kernel, bias)。"""
    kernel = conv.weight.clone()
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std
def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
) -> nn.Sequential:
    """Conv1d + BatchNorm1d 组合。"""
    return nn.Sequential(
        nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False,
        ),
        nn.BatchNorm1d(out_channels),
    )
class ReparamLargeKernelConv(nn.Module):
    """大卷积核重参数化卷积（深度可分离）。
    【groups 语义】深度卷积分支 groups = in_channels，
    要求 out_channels 是 in_channels 的整数倍。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        small_kernel: int,
        stride: int = 1,
        groups: int | None = None,
        small_kernel_merged: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 默认按逐通道深度卷积
        self.groups = in_channels if groups is None else groups
        self.small_kernel_merged = small_kernel_merged
        self.lkb_origin = conv_bn(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, groups=self.groups,
        )
        if small_kernel_merged:
            self.small_conv = None
        else:
            self.small_conv = conv_bn(
                in_channels, out_channels, small_kernel,
                stride=stride, padding=small_kernel // 2, groups=self.groups,
            )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.lkb_origin(inputs)
        if self.small_conv is not None:
            out = out + self.small_conv(inputs)
        return out
    def get_equivalent_kernel_bias(self):
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
        """把双分支卷积 + BN 合并成单个 Conv1d。"""
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        merged = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=self.groups,
            bias=True,
        )
        merged.weight.data.copy_(eq_k)
        merged.bias.data.copy_(eq_b)
        self.lkb_origin = merged
        self.small_conv = None
# ============================================================
# Block / Stage / Backbone
# ============================================================
class Block(nn.Module):
    """ModernTCN Block：大核深度卷积 + 逐点卷积 + FFN。"""
    def __init__(
        self,
        large_size: int,
        small_size: int,
        dim: int,
        dw_dims: int,
        small_kernel_merged: bool = False,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = ReparamLargeKernelConv(
            in_channels=dim,
            out_channels=dw_dims,
            kernel_size=large_size,
            small_kernel=small_size,
            stride=1,
            groups=dim,          # 逐通道深度卷积
            small_kernel_merged=small_kernel_merged,
        )
        self.norm = nn.BatchNorm1d(dw_dims)
        self.pwconv1 = nn.Linear(dw_dims, dw_dims)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dw_dims, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0 else None
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim, L]
        shortcut = x
        x = self.dwconv(x)          # [B, dw_dims, L]
        x = self.norm(x)
        x = x.permute(0, 2, 1)      # [B, L, dw_dims]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)         # [B, L, dim]
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)      # [B, dim, L]
        return shortcut + x
class Stage(nn.Module):
    """多 Block 堆叠 + 可选下采样。"""
    def __init__(
        self,
        num_blocks: int,
        large_size: int,
        small_size: int,
        dims: int,
        dw_dims: int,
        downsample_ratio: int,
        small_kernel_merged: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(
                large_size=large_size,
                small_size=small_size,
                dim=dims,
                dw_dims=dw_dims,
                small_kernel_merged=small_kernel_merged,
            )
            for _ in range(num_blocks)
        ])
        if downsample_ratio > 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(dims, dims, kernel_size=1, stride=1),
                nn.BatchNorm1d(dims),
            )
        else:
            self.downsample = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.blocks(x))
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
        self.dims = config.dims
        # patch 数量：T -> patch_num
        self.patch_num = int((self.seq_len - self.patch_size) / self.patch_stride + 1)
        # stem：[B, C, T] -> [B, dim0, patch_num]
        stem_dim = int(self.dims[0] * config.stem_ratio)
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.enc_in, stem_dim,
                kernel_size=self.patch_size,
                stride=self.patch_stride,
                bias=False,
            ),
            nn.BatchNorm1d(stem_dim),
        )
        # 4 个 Stage
        self.stages = nn.Sequential(*[
            Stage(
                num_blocks=config.num_blocks[i],
                large_size=config.large_size[i],
                small_size=config.small_size[i],
                dims=config.dims[i],
                dw_dims=config.dw_dims[i],
                downsample_ratio=config.downsample_ratio,
                small_kernel_merged=config.small_kernel_merged,
            )
            for i in range(len(config.num_blocks))
        ])
        # 预测头：[B, C, dim, patch_num] -> [B, C, pred_len]
        self.head_nf = self.dims[-1] * self.patch_num
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),          # [B, C, dim*patch_num]
            nn.Linear(self.head_nf, self.pred_len),
            nn.Dropout(config.head_dropout),
        )
        # RevIN
        self.revin = RevIN(
            num_features=self.enc_in,
            affine=config.affine,
            subtract_last=config.subtract_last,
        ) if config.revin else None
    def forward(self, x: torch.Tensor, te: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, C, T] -> y: [B, pred_len, C]"""
        mean = None
        stdev = None
        # 1) RevIN 归一化
        if self.revin is not None:
            x, mean, stdev = self.revin.normalize(x)   # [B, C, T]
        # 2) 主干特征
        x = self.stem(x)        # [B, dim0, patch_num]
        x = self.stages(x)      # [B, dim, patch_num]
        B, dim, patch_num = x.shape
        # 3) 每个变量共享卷积特征（individual=False）
        x = x.unsqueeze(1).expand(B, self.enc_in, dim, patch_num)
        # 4) 预测头
        x = self.head(x)        # [B, C, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, C]
        # 5) RevIN 反归一化
        if self.revin is not None:
            x = self.revin.denormalize(x, mean, stdev)
        return x
    def structural_reparam(self) -> None:
        """结构重参数化：合并大核+小核卷积分支。"""
        for m in self.modules():
            if isinstance(m, ReparamLargeKernelConv):
                m.merge_kernel()
class ModernTCN(nn.Module):
    """ModernTCN 完整模型。"""
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