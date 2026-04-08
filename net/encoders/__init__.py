"""Encoder exports for few-shot learning models."""

from .base_encoder import Conv64F_Encoder, get_norm_layer
from .protonet_encoder import Conv64F_Paper_Encoder
from .matchingnet_encoder import MatchingNetEncoder
from .relationnet_encoder import RelationNetEncoder
from .slim_mamba_encoder import SlimMambaEncoder
from .smnet_conv64f_encoder import SMNetConv64FEncoder, build_resnet12_family_encoder

try:
    from .resnet12_encoder import ResNet12Encoder
except Exception:  # pragma: no cover - optional dependency path
    ResNet12Encoder = None

try:
    from .resnet18_encoder import ResNet18Encoder
except Exception:  # pragma: no cover - optional dependency path
    ResNet18Encoder = None

__all__ = [
    'Conv64F_Encoder',
    'Conv64F_Paper_Encoder',
    'MatchingNetEncoder',
    'RelationNetEncoder',
    'SlimMambaEncoder',
    'SMNetConv64FEncoder',
    'ResNet12Encoder',
    'ResNet18Encoder',
    'build_resnet12_family_encoder',
    'get_norm_layer',
]
