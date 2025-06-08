import torch
import pytest
from dataclasses import dataclass
import escnn
from models.orbitgrasp_unet import EquiformerUnet

from escnn.group import SO3
from escnn.gspaces import no_base_space


@pytest.fixture()
def so3_group():
    so3 = no_base_space(SO3())
    return so3


@dataclass
class PointCloud:
    b: torch.Tensor  # batch indices
    x: torch.Tensor  # node coordinates
    n: torch.Tensor  # node features/normals


@pytest.fixture
def single_point_cloud():
    """Fixture providing a single point cloud for testing."""
    # Create a small point cloud with 10 points
    num_points = 10
    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Batch indices (all points belong to batch 0)
    b = torch.zeros(num_points, dtype=torch.long, device=device)

    # Random 3D coordinates
    x = torch.randn(num_points, 3, device=device)

    # Random normal vectors (normalized)
    n = torch.randn(num_points, 3, device=device)
    n = n / torch.norm(n, dim=1, keepdim=True)

    return PointCloud(b=b, x=x, n=n)


@pytest.fixture
def batched_point_cloud():
    """Fixture providing multiple point clouds for testing."""
    # Create two point clouds with different numbers of points
    num_points_1 = 10
    num_points_2 = 15
    batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First point cloud
    b1 = torch.zeros(num_points_1, dtype=torch.long, device=device)
    x1 = torch.randn(num_points_1, 3, device=device)
    n1 = torch.randn(num_points_1, 3, device=device)
    n1 = n1 / torch.norm(n1, dim=1, keepdim=True)

    # Second point cloud
    b2 = torch.ones(num_points_2, dtype=torch.long, device=device)
    x2 = torch.randn(num_points_2, 3, device=device)
    n2 = torch.randn(num_points_2, 3, device=device)
    n2 = n2 / torch.norm(n2, dim=1, keepdim=True)

    return [PointCloud(b=b1, x=x1, n=n1), PointCloud(b=b2, x=x2, n=n2)]


@pytest.fixture
def model_fp32():
    """Fixture providing the model in fp32 precision."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EquiformerUnet(
        max_neighbors=(18, 1000, 1000),
        max_radius=(0.015, 0.03, 0.08),
        pool_ratio=(1, 0.3, 0.3),
        num_layers=(2, 3, 2),
        sphere_channels=(64, 64, 64),
        attn_hidden_channels=(64, 64, 64),
        num_heads=4,
        attn_alpha_channels=(64, 64, 64),
        attn_value_channels=(16, 16, 16),
        ffn_hidden_channels=(128, 128, 128),
        lmax_list=[3],
        mmax_list=[2],
        edge_channels=(48, 48, 48),
        num_distance_basis=(256, 256, 256),
    )
    return model.to(device)


@pytest.fixture
def model_fp16():
    """Fixture providing the model in fp16 mixed precision."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EquiformerUnet(
        max_neighbors=(18, 1000, 1000),
        max_radius=(0.015, 0.03, 0.08),
        pool_ratio=(1, 0.3, 0.3),
        num_layers=(2, 3, 2),
        sphere_channels=(64, 64, 64),
        attn_hidden_channels=(64, 64, 64),
        num_heads=4,
        attn_alpha_channels=(64, 64, 64),
        attn_value_channels=(16, 16, 16),
        ffn_hidden_channels=(128, 128, 128),
        lmax_list=[3],
        mmax_list=[2],
        edge_channels=(48, 48, 48),
        num_distance_basis=(256, 256, 256),
        mixed_precision="fp16",
    )
    return model.to(device)


@pytest.fixture
def model_bf16():
    """Fixture providing the model in bf16 mixed precision."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EquiformerUnet(
        max_neighbors=(18, 1000, 1000),
        max_radius=(0.015, 0.03, 0.08),
        pool_ratio=(1, 0.3, 0.3),
        num_layers=(2, 3, 2),
        sphere_channels=(64, 64, 64),
        attn_hidden_channels=(64, 64, 64),
        num_heads=4,
        attn_alpha_channels=(64, 64, 64),
        attn_value_channels=(16, 16, 16),
        ffn_hidden_channels=(128, 128, 128),
        lmax_list=[3],
        mmax_list=[2],
        edge_channels=(48, 48, 48),
        num_distance_basis=(256, 256, 256),
        mixed_precision="bf16",
    )
    return model.to(device)
