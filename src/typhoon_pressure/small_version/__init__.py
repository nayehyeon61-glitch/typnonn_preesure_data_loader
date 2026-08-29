"""Small dual-objective model for long-range distribution and local track learning."""

from .config import DualLossConfig, EastAsiaBounds, SmallModelConfig
from .dataset import DualTargetDataset, SpatialDistributionLookup
from .losses import DualObjectiveLoss
from .model import SmallDualScaleModel

__all__ = [
    "DualLossConfig",
    "DualObjectiveLoss",
    "DualTargetDataset",
    "EastAsiaBounds",
    "SmallDualScaleModel",
    "SmallModelConfig",
    "SpatialDistributionLookup",
]

