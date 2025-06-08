import torch
import pytest
from tests.fixtures import (
    single_point_cloud,
    batched_point_cloud,
    model_fp16,
    model_bf16,
)


def test_forward_fp16(single_point_cloud, model_fp16):
    """Test forward pass with fp16 mixed precision."""
    # Run forward pass
    output = model_fp16(single_point_cloud)

    # Check output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.device == model_fp16.device  # Check output is on correct device
    assert output.dim() == 2  # Should be [num_points, 1]
    assert (
        output.shape[0] == single_point_cloud.x.shape[0]
    )  # Number of points should match input
    assert output.shape[1] == 1  # Single output value per point


def test_forward_bf16(single_point_cloud, model_bf16):
    """Test forward pass with bf16 mixed precision."""
    # Run forward pass
    output = model_bf16(single_point_cloud)

    # Check output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.device == model_bf16.device  # Check output is on correct device
    assert output.dim() == 2  # Should be [num_points, 1]
    assert (
        output.shape[0] == single_point_cloud.x.shape[0]
    )  # Number of points should match input
    assert output.shape[1] == 1  # Single output value per point


def test_forward_fp16_batched(batched_point_cloud, model_fp16):
    """Test forward pass with fp16 mixed precision on batched data."""
    # Run forward pass
    output = model_fp16(batched_point_cloud)

    # Check output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.device == model_fp16.device  # Check output is on correct device
    assert output.dim() == 2  # Should be [total_points, 1]

    # Calculate total number of points across all point clouds
    total_points = sum(pc.x.shape[0] for pc in batched_point_cloud)
    assert (
        output.shape[0] == total_points
    )  # Number of points should match total input points
    assert output.shape[1] == 1  # Single output value per point


def test_forward_bf16_batched(batched_point_cloud, model_bf16):
    """Test forward pass with bf16 mixed precision on batched data."""
    # Run forward pass
    output = model_bf16(batched_point_cloud)

    # Check output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.device == model_bf16.device  # Check output is on correct device
    assert output.dim() == 2  # Should be [total_points, 1]

    # Calculate total number of points across all point clouds
    total_points = sum(pc.x.shape[0] for pc in batched_point_cloud)
    assert (
        output.shape[0] == total_points
    )  # Number of points should match total input points
    assert output.shape[1] == 1  # Single output value per point
