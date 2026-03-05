"""XResNet1D building blocks - ported from ecg_ptbxl_benchmarking (no FastAI dependency)."""

from enum import Enum
import torch
import torch.nn as nn

# NormType for ConvLayer/ResBlock
NormType = Enum("NormType", "Batch BatchZero Weight Spectral Instance InstanceZero")


def _conv_func(ndim=2, transpose=False):
    """Return the proper conv ndim function."""
    assert 1 <= ndim <= 3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')


def init_default(m, func=nn.init.kaiming_normal_):
    """Initialize m weights with func and set bias to 0."""
    if func and hasattr(m, "weight"):
        func(m.weight)
    with torch.no_grad():
        if getattr(m, "bias", None) is not None:
            m.bias.fill_(0.0)
    return m


def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    """Norm layer with nf features and ndim."""
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0.0 if zero else 1.0)
    return bn


def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    """BatchNorm layer."""
    return _get_norm("BatchNorm", nf, ndim, zero=norm_type == NormType.BatchZero, **kwargs)


def InstanceNorm(nf, ndim=2, norm_type=NormType.Instance, **kwargs):
    """InstanceNorm layer."""
    return _get_norm("InstanceNorm", nf, ndim, zero=norm_type == NormType.InstanceZero, **kwargs)


class ConvLayer(nn.Sequential):
    """Conv (ni->nf), ReLU, norm. Supports Batch and Instance norm only (no weight/spectral)."""

    def __init__(
        self,
        ni,
        nf,
        ks=3,
        stride=1,
        padding=None,
        bias=None,
        ndim=2,
        norm_type=NormType.Batch,
        bn_1st=True,
        act_cls=nn.ReLU,
        transpose=False,
        init=nn.init.kaiming_normal_,
        xtra=None,
        **kwargs,
    ):
        if padding is None:
            padding = (ks - 1) // 2 if not transpose else 0
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None:
            bias = not (bn or inn)
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = init_default(
            conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs),
            init,
        )
        layers = [conv]
        act_bn = []
        if act_cls is not None:
            act_bn.append(act_cls())
        if bn:
            act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn:
            act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st:
            act_bn.reverse()
        layers += act_bn
        if xtra is not None:
            layers.append(xtra)
        super().__init__(*layers)


def AdaptiveAvgPool(sz=1, ndim=2):
    """AdaptiveAvgPool for ndim."""
    assert 1 <= ndim <= 3
    return getattr(nn, f"AdaptiveAvgPool{ndim}d")(sz)


def MaxPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    """MaxPool for ndim."""
    assert 1 <= ndim <= 3
    return getattr(nn, f"MaxPool{ndim}d")(ks, stride=stride, padding=padding)


def AvgPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    """AvgPool for ndim."""
    assert 1 <= ndim <= 3
    return getattr(nn, f"AvgPool{ndim}d")(
        ks, stride=stride, padding=padding, ceil_mode=ceil_mode
    )


class AdaptiveConcatPool1d(nn.Module):
    """Concatenates AdaptiveAvgPool1d and AdaptiveMaxPool1d along channels."""

    def __init__(self, sz=None):
        super().__init__()
        sz = sz or 1
        self.ap = nn.AdaptiveAvgPool1d(sz)
        self.mp = nn.AdaptiveMaxPool1d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)


class ResBlock(nn.Module):
    """ResNet block from ni to nf. No SEModule or SelfAttention (reduction=None, sa=False)."""

    def __init__(
        self,
        expansion,
        ni,
        nf,
        stride=1,
        kernel_size=3,
        groups=1,
        reduction=None,
        nh1=None,
        nh2=None,
        dw=False,
        g2=1,
        sa=False,
        sym=False,
        norm_type=NormType.Batch,
        act_cls=nn.ReLU,
        ndim=2,
        pool=AvgPool,
        pool_first=True,
        **kwargs,
    ):
        super().__init__()
        norm2 = (
            NormType.BatchZero
            if norm_type == NormType.Batch
            else NormType.InstanceZero
            if norm_type == NormType.Instance
            else norm_type
        )
        nh2 = nf if nh2 is None else nh2
        nh1 = nh2 if nh1 is None else nh1
        nf, ni = nf * expansion, ni * expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
        if expansion == 1:
            layers = [
                ConvLayer(ni, nh2, kernel_size, stride=stride, groups=ni if dw else groups, **k0),
                ConvLayer(nh2, nf, kernel_size, groups=g2, **k1),
            ]
        else:
            layers = [
                ConvLayer(ni, nh1, 1, **k0),
                ConvLayer(nh1, nh2, kernel_size, stride=stride, groups=nh1 if dw else groups, **k0),
                ConvLayer(nh2, nf, 1, groups=g2, **k1),
            ]
        self.convs = nn.Sequential(*layers)
        self.convpath = nn.Sequential(self.convs)
        idpath = []
        if ni != nf:
            idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride != 1:
            idpath.insert(1 if pool_first else 0, pool(2, ndim=ndim, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = nn.ReLU(inplace=True) if act_cls is nn.ReLU else act_cls()

    def forward(self, x):
        return self.act(self.convpath(x) + self.idpath(x))


def init_cnn(m):
    """Initialize CNN layers with Kaiming init."""
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for child in m.children():
        init_cnn(child)
