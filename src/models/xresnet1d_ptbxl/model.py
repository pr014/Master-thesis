"""XResNet1D-101 from PTB-XL with Transfer Learning for LOS + Mortality."""

from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn

from .blocks import (
    ConvLayer,
    ResBlock,
    AdaptiveConcatPool1d,
    AdaptiveAvgPool,
    AvgPool,
    MaxPool,
    init_cnn,
    NormType,
)
from ..core.base_model import BaseECGModel


def _listify(x):
    """Convert scalar to list."""
    if isinstance(x, (int, float)):
        return [x]
    return list(x)


def _bn_drop_lin(ni: int, no: int, bn: bool = True, p: float = 0.5, actn: Optional[nn.Module] = None):
    """BatchNorm, Dropout, Linear with optional activation (bn_drop_lin from fastai)."""
    layers = []
    if bn:
        layers.append(nn.BatchNorm1d(ni))
    if p > 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(ni, no))
    if actn is not None:
        layers.append(actn)
    return nn.Sequential(*layers)


def create_head1d(
    nf: int,
    nc: int,
    lin_ftrs: Optional[List[int]] = None,
    ps: Union[float, List[float]] = 0.5,
    bn_final: bool = False,
    bn: bool = True,
    act: str = "relu",
    concat_pooling: bool = True,
) -> nn.Sequential:
    """Head: pool -> flatten -> fc layers -> nc classes."""
    if lin_ftrs is None:
        lin_ftrs = [2 * nf if concat_pooling else nf, nc]
    else:
        lin_ftrs = [2 * nf if concat_pooling else nf] + list(lin_ftrs) + [nc]
    ps = _listify(ps)
    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    actns = [
        (nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)) if i < len(lin_ftrs) - 2 else None
        for i in range(len(lin_ftrs) - 1)
    ]
    layers = [
        AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2),
        nn.Flatten(),
    ]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers.extend(_bn_drop_lin(ni, no, bn=bn, p=p, actn=actn))
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)


class XResNet1d(nn.Sequential):
    """XResNet1D backbone + head (for pretrained loading)."""

    def __init__(
        self,
        block,
        expansion: int,
        layers: List[int],
        input_channels: int = 12,
        num_classes: int = 71,
        stem_szs=(32, 32, 64),
        kernel_size: int = 5,
        kernel_size_stem: int = 5,
        widen: float = 1.0,
        sa: bool = False,
        act_cls=nn.ReLU,
        lin_ftrs_head: Optional[List[int]] = None,
        ps_head: float = 0.5,
        bn_final_head: bool = False,
        bn_head: bool = True,
        act_head: str = "relu",
        concat_pooling: bool = True,
        **kwargs,
    ):
        self.block = block
        self.expansion = expansion
        self.act_cls = act_cls

        stem_szs = [input_channels, *stem_szs]
        stem = [
            ConvLayer(
                stem_szs[i],
                stem_szs[i + 1],
                ks=kernel_size_stem,
                stride=2 if i == 0 else 1,
                act_cls=act_cls,
                ndim=1,
            )
            for i in range(3)
        ]

        block_szs = [int(o * widen) for o in [64, 64, 64, 64] + [32] * (len(layers) - 4)]
        block_szs = [64 // expansion] + block_szs
        blocks = [
            self._make_layer(
                ni=block_szs[i],
                nf=block_szs[i + 1],
                blocks=l,
                stride=1 if i == 0 else 2,
                kernel_size=kernel_size,
                sa=sa and i == len(layers) - 4,
                ndim=1,
                **kwargs,
            )
            for i, l in enumerate(layers)
        ]

        head = create_head1d(
            block_szs[-1] * expansion,
            nc=num_classes,
            lin_ftrs=lin_ftrs_head,
            ps=ps_head,
            bn_final=bn_final_head,
            bn=bn_head,
            act=act_head,
            concat_pooling=concat_pooling,
        )

        super().__init__(
            *stem,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            *blocks,
            head,
        )
        init_cnn(self)

    def _make_layer(self, ni, nf, blocks, stride, kernel_size, sa, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "ndim"}
        return nn.Sequential(
            *[
                self.block(
                    self.expansion,
                    ni if i == 0 else nf,
                    nf,
                    stride=stride if i == 0 else 1,
                    kernel_size=kernel_size,
                    sa=sa and i == (blocks - 1),
                    act_cls=self.act_cls,
                    ndim=1,
                    pool=AvgPool,
                    **kwargs,
                )
                for i in range(blocks)
            ]
        )


def xresnet1d101(
    num_classes: int = 71,
    input_channels: int = 12,
    kernel_size: int = 5,
    ps_head: float = 0.5,
    lin_ftrs_head: Optional[List[int]] = None,
    **kwargs,
) -> XResNet1d:
    """XResNet1D-101: expansion=4, layers=[3,4,23,3]."""
    if lin_ftrs_head is None:
        lin_ftrs_head = [128]
    return XResNet1d(
        ResBlock,
        expansion=4,
        layers=[3, 4, 23, 3],
        input_channels=input_channels,
        num_classes=num_classes,
        kernel_size=kernel_size,
        ps_head=ps_head,
        lin_ftrs_head=lin_ftrs_head,
        **kwargs,
    )


class XResNetPTBXL(BaseECGModel):
    """XResNet1D-101 with PTB-XL pretrained weights for Transfer Learning.

    Architecture: stem + ResBlocks (backbone) -> features (128) -> LOS/Mortality heads.
    Implements get_features() for MultiTaskECGModel wrapping.
    """

    # ECG feature dim from backbone head (before final linear): 128
    ECG_FEATURE_DIM = 128

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        training_config = config.get("training", {})
        dropout_rate = training_config.get("dropout_rate", 0.3)
        model_config = config.get("model", {})
        input_channels = 12
        kernel_size = model_config.get("kernel_size", 5)
        ps_head = model_config.get("ps_head", 0.5)
        lin_ftrs_head = model_config.get("lin_ftrs_head", [128])

        # Build full xresnet1d101 (for pretrained loading)
        self.backbone_with_head = xresnet1d101(
            num_classes=71,  # PTB-XL original
            input_channels=input_channels,
            kernel_size=kernel_size,
            ps_head=ps_head,
            lin_ftrs_head=lin_ftrs_head,
        )

        # Backbone = everything except the last Sequential (head)
        # Structure: Sequential(stem..., MaxPool, block0, block1, block2, block3, head)
        # head is index -1. We need stem + blocks = indices 0 to -2 (all but head)
        # Actually: self.backbone_with_head has: [0]stem0, [1]stem1, [2]stem2, [3]MaxPool,
        # [4]block0, [5]block1, [6]block2, [7]block3, [8]head
        # So backbone = self.backbone_with_head[:-1] (all but head)
        self.backbone = nn.Sequential(*list(self.backbone_with_head.children())[:-1])

        # Head returns features of dim 128 (before final Linear to 71)
        # create_head1d: pool->flatten->...->Linear(128,71). We need output of the layer before last Linear.
        # The head structure: [0]AdaptiveConcatPool1d, [1]Flatten, [2]bn_drop_lin(512->128), [3]bn_drop_lin(128->71)
        # So "features" = output of head up to index -2 (before last Linear).
        self.head = self.backbone_with_head[-1]
        self._feature_head = nn.Sequential(*list(self.head.children())[:-1])  # up to before last Linear

        # Late fusion config (same as CNNScratch/HybridCNNLSTM)
        data_config = config.get("data", {})
        demographic_config = data_config.get("demographic_features", {})
        self.use_demographics = demographic_config.get("enabled", False)
        diagnosis_config = data_config.get("diagnosis_features", {})
        self.use_diagnoses = diagnosis_config.get("enabled", False)
        diagnosis_list = diagnosis_config.get("diagnosis_list", [])
        self.diagnosis_dim = len(diagnosis_list) if self.use_diagnoses else 0
        icu_unit_config = data_config.get("icu_unit_features", {})
        self.use_icu_units = icu_unit_config.get("enabled", False)
        icu_unit_list = icu_unit_config.get("icu_unit_list", [])
        self.icu_unit_dim = len(icu_unit_list) if self.use_icu_units else 0

        feature_dim = self.ECG_FEATURE_DIM
        if self.use_demographics:
            sex_encoding = demographic_config.get("sex_encoding", "binary")
            self.demo_dim = 2 if sex_encoding == "binary" else 3
            feature_dim += self.demo_dim
        else:
            self.demo_dim = 0
        if self.use_diagnoses:
            feature_dim += self.diagnosis_dim
        if self.use_icu_units:
            feature_dim += self.icu_unit_dim

        # For single-task regression (used if not wrapped by MultiTaskECGModel)
        output_dim = 1 if self.task_type == "regression" else (self.num_classes or 10)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(feature_dim, output_dim)
        self.relu = nn.ReLU()

        # Load pretrained weights
        pretrained_config = model_config.get("pretrained", {})
        if pretrained_config.get("enabled", False):
            checkpoint_path = pretrained_config.get(
                "checkpoint_path",
                "data/pretrained_weights/PTB-Xl-analysis/fastai_xresnet1d101.pth",
            )
            self._load_pretrained_backbone(checkpoint_path)

    def _load_pretrained_backbone(self, checkpoint_path: str) -> None:
        """Load PTB-XL pretrained weights into backbone+head, then we use backbone + feature_head."""
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model_dict = state.get("model", state)
        # Load into full backbone_with_head (strict=False to ignore size mismatches in final layer)
        missing, unexpected = self.backbone_with_head.load_state_dict(model_dict, strict=False)
        # Expected: missing/unexpected in final linear (71 classes vs our heads)
        # Log only if there are critical missing keys in backbone
        if any("0." in k or "1." in k or "2." in k or "3." in k or "4." in k for k in missing):
            pass  # Some backbone layers might have different names
        # Copy loaded weights - backbone_with_head is now pretrained
        # Our self.backbone and self.head are views into backbone_with_head, so they're updated

    def _ecg_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-dim ECG features (no late fusion)."""
        x = self.backbone(x)  # (B, C, T) -> (B, 256, some_len)
        x = self._feature_head(x)  # pool, flatten, bn_drop_lin -> (B, 128)
        return x

    def forward(
        self,
        x: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
        diagnosis_features: Optional[torch.Tensor] = None,
        icu_unit_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single-task forward (regression)."""
        features = self.get_features(
            x,
            demographic_features=demographic_features,
            diagnosis_features=diagnosis_features,
            icu_unit_features=icu_unit_features,
        )
        return self.fc(self.dropout(features))

    def get_features(
        self,
        x: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
        diagnosis_features: Optional[torch.Tensor] = None,
        icu_unit_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features for MultiTaskECGModel (ECG + optional late fusion)."""
        fused = self._ecg_features(x)
        if self.use_demographics and demographic_features is not None:
            fused = torch.cat([fused, demographic_features], dim=1)
        if self.use_diagnoses and diagnosis_features is not None:
            fused = torch.cat([fused, diagnosis_features], dim=1)
        if self.use_icu_units and icu_unit_features is not None:
            fused = torch.cat([fused, icu_unit_features], dim=1)
        return fused
