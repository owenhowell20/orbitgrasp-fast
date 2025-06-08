import torch
import math
from typing import Optional
import warnings
import torch.nn.functional as F


@torch.jit.script
def gaussian(x, mean, std):
    # pi = 3.14159
    # a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2))  # / (a * std)


@torch.jit.script
def soft_step(x, n: int = 3):
    return (x > 0) * ((x < 1) * ((n + 1) * x.pow(n) - n * x.pow(n + 1)) + (x >= 1))


@torch.jit.script
def soft_cutoff(x, thr: float = 0.8, n: int = 3):
    x = (x - thr) / (1 - thr)
    return 1 - soft_step(x, n=n)


@torch.jit.script
def soft_square_cutoff(
    x, thr: float = 0.8, n: int = 3, infinite: bool = False
) -> torch.Tensor:
    if infinite:
        return soft_cutoff(x, thr=thr, n=n) * (x > 0.5) + soft_cutoff(
            1 - x, thr=thr, n=n
        ) * (x <= 0.5)
    else:
        return (x > 0.5) + soft_cutoff(1 - x, thr=thr, n=n) * (x <= 0.5)


# From Graphormer
class GaussianRadialBasisLayer(torch.nn.Module):
    def __init__(self, num_basis, cutoff):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0
        self.mean = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        self.std_init_max = 1.0
        self.std_init_min = 1.0 / self.num_basis
        self.mean_init_max = 1.0
        self.mean_init_min = 0
        torch.nn.init.uniform_(self.mean, self.mean_init_min, self.mean_init_max)
        torch.nn.init.uniform_(self.std, self.std_init_min, self.std_init_max)
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, dist):
        x = dist / self.cutoff
        x = x.unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, self.num_basis)
        mean = self.mean
        std = self.std.abs() + 1e-5
        x = gaussian(x, mean, std)
        return x

    def extra_repr(self):
        return "mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}".format(
            self.mean_init_max, self.mean_init_min, self.std_init_max, self.std_init_min
        )


# From Diffusion-EDF
class GaussianRadialBasisLayerFiniteCutoff(torch.nn.Module):
    def __init__(
        self,
        num_basis: int,
        cutoff: float,
        soft_cutoff: bool = True,
        offset: Optional[float] = None,
        cutoff_thr_ratio: float = 0.8,
        infinite: bool = False,
    ):
        super().__init__()
        self.num_basis: int = num_basis
        self.cutoff: float = float(cutoff)
        if offset is None:
            offset = (
                0.01 * self.cutoff
            )  # For stability, weights should be zero when edge length is very small (otherwise, gradients of spherical harmonics would blow up).
        self.offset: float = float(offset)
        if self.offset < 0.0:
            warnings.warn(
                f"Negative offset ({self.offset}) is provided for radial basis encoder. Are you sure?"
            )

        self.mean_init_max = 1.0
        self.mean_init_min = 0
        mean = torch.linspace(
            self.mean_init_min, self.mean_init_max, self.num_basis + 2
        )[1:-1].unsqueeze(0)
        self.mean = torch.nn.Parameter(mean)

        self.std_logit = torch.nn.Parameter(
            torch.zeros(1, self.num_basis)
        )  # Softplus logit
        self.weight_logit = torch.nn.Parameter(
            torch.zeros(1, self.num_basis)
        )  # Sigmoid logit

        init_std = 2.0 / self.num_basis
        torch.nn.init.constant_(
            self.std_logit, math.log(math.exp((init_std)) - 1)
        )  # Inverse Softplus

        self.max_weight = 4.0
        torch.nn.init.constant_(
            self.weight_logit, -math.log(self.max_weight / 1.0 - 1)
        )  # Inverse Softplus

        self.soft_cutoff: bool = soft_cutoff
        self.cutoff_thr_ratio: float = cutoff_thr_ratio
        assert cutoff_thr_ratio <= 0.95

        self.normalizer = math.sqrt(self.num_basis)
        self.infinite = infinite

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = (dist - self.offset) / (self.cutoff - self.offset)
        dist = dist.unsqueeze(-1)

        x = dist.expand(-1, self.num_basis)
        mean = self.mean
        std = F.softplus(self.std_logit) + 1e-5
        x = gaussian(x, mean, std)
        x = torch.sigmoid(self.weight_logit) * self.max_weight * x

        if self.soft_cutoff is True:
            x = x * soft_square_cutoff(
                dist, thr=self.cutoff_thr_ratio, infinite=self.infinite
            )

        return x * self.normalizer
