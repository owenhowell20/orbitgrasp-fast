from typing import NamedTuple
import torch


class FeaturedPoints(NamedTuple):
    x: torch.Tensor  # Position
    n: torch.Tensor  # Normal
    b: torch.Tensor  # Batch idx
