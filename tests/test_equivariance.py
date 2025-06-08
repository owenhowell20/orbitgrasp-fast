import torch
import pytest
import escnn
from tests.fixtures import single_point_cloud, model_fp32, model_fp16, model_bf16
from models.orbitgrasp_unet import EquiformerUnet

# def test_model_equivariant(so3_group, single_point_cloud):

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = EquiformerUnet(
#         max_neighbors=(18, 1000, 1000),
#         max_radius=(0.015, 0.03, 0.08),
#         pool_ratio=(1, 0.3, 0.3),
#         num_layers=(2, 3, 2),
#         sphere_channels=(64, 64, 64),
#         attn_hidden_channels=(64, 64, 64),
#         num_heads=4,
#         attn_alpha_channels=(64, 64, 64),
#         attn_value_channels=(16, 16, 16),
#         ffn_hidden_channels=(128, 128, 128),
#         lmax_list=[3],
#         mmax_list=[2],
#         edge_channels=(48, 48, 48),
#         num_distance_basis=(256, 256, 256),
#     )

#     # Move model to device
#     model = model.to(device)

#     # Run forward pass
#     output = model(single_point_cloud)


#     invariant_rep = so3_group.irrep(0)
#     vector_rep = so3_group.irrep(1)

#     ## Field types
#     invariant_type = escnn.nn.FieldType(
#         so3_group, invariant_rep
#     )
#     vector_type = escnn.nn.FieldType(
#         so3_group, vector_rep
#     )

#     ### assuming that features are invariant, apply rotation to node positions of point cloud,

#     ...


#     # ### output tensor
#     # v_o = project_tensor_product(q, k, ell_out, type="inter")

#     # ### NOTE: MUST PERMUTE FIRST, ( i.e. (b,N, m_l,d_l) format ) then reshape !!!!!!
#     # q = q.permute(0, 1, 3, 2).reshape(
#     #     batch_size * num_tokens, (2 * ell_in_1 + 1) * m1
#     # )
#     # k = k.permute(0, 1, 3, 2).reshape(
#     #     batch_size * num_tokens, (2 * ell_in_2 + 1) * m2
#     # )
#     # v_o = v_o.permute(0, 1, 3, 2).reshape(
#     #     batch_size * num_tokens, (2 * ell_out + 1) * m1 * m2
#     # )


#     # # Wrap tensors
#     # q = escnn.nn.GeometricTensor(q, type_ell_in_1)
#     # k = escnn.nn.GeometricTensor(k, type_ell_in_2)
#     # v_o = escnn.nn.GeometricTensor(v_o, type_out)

#     # ### apply G transformation
#     # g = so3_group.fibergroup.sample()

#     # # Apply the transformation to the vector features (x)
#     # q_g = q.transform(g).tensor
#     # k_g = k.transform(g).tensor
#     # v_o_g = v_o.transform(g).tensor

#     # ### now, convert back
#     # q_g = q_g.reshape(
#     #     batch_size, num_tokens, m1, 2 * ell_in_1 + 1
#     # ).permute(0, 1, 3, 2)
#     # k_g = k_g.reshape(
#     #     batch_size, num_tokens, m2, 2 * ell_in_2 + 1
#     # ).permute(0, 1, 3, 2)
#     # v_o_g = v_o_g.reshape(
#     #     batch_size, num_tokens, m1 * m2, 2 * ell_out + 1
#     # ).permute(0, 1, 3, 2)

#     # v_g = project_tensor_product(
#     #     q_g, k_g, ell_out, type="inter"
#     # )

#     # assert v_o_g.shape == v_g.shape, "Shape mismatch"
#     # assert torch.allclose(v_o_g, v_g, atol=1e-2)


def test_model_equivariant_multiple_rotations(
    so3_group, single_point_cloud, model_fp32
):
    """Test equivariance with multiple random rotations."""
    # Run forward pass on original point cloud
    output_original = model_fp32(single_point_cloud)

    # Test multiple random rotations
    num_rotations = 5
    for _ in range(num_rotations):
        # Create a random rotation
        rotation = so3_group.sample()

        # Rotate the point cloud positions (x) and normals (n)
        rotated_pcd = single_point_cloud
        rotated_pcd.x = torch.matmul(rotation, single_point_cloud.x.T).T
        rotated_pcd.n = torch.matmul(rotation, single_point_cloud.n.T).T

        # Run forward pass on rotated point cloud
        output_rotated = model_fp32(rotated_pcd)

        # Check that outputs are on the same device
        assert output_original.device == output_rotated.device

        # Check that the outputs have the same shape
        assert output_original.shape == output_rotated.shape

        # Since the model should be equivariant, the outputs should be equal
        # (up to numerical precision)
        torch.testing.assert_close(
            output_original, output_rotated, rtol=1e-3, atol=1e-3
        )


def test_model_equivariant_multiple_rotations_fp16(
    so3_group, single_point_cloud, model_fp16
):
    """Test equivariance with multiple random rotations using fp16 mixed precision."""
    # Run forward pass on original point cloud
    output_original = model_fp16(single_point_cloud)

    # Test multiple random rotations
    num_rotations = 5
    for _ in range(num_rotations):
        # Create a random rotation
        rotation = so3_group.sample()

        # Rotate the point cloud positions (x) and normals (n)
        rotated_pcd = single_point_cloud
        rotated_pcd.x = torch.matmul(rotation, single_point_cloud.x.T).T
        rotated_pcd.n = torch.matmul(rotation, single_point_cloud.n.T).T

        # Run forward pass on rotated point cloud
        output_rotated = model_fp16(rotated_pcd)

        # Check that outputs are on the same device
        assert output_original.device == output_rotated.device

        # Check that the outputs have the same shape
        assert output_original.shape == output_rotated.shape

        # Since the model should be equivariant, the outputs should be equal
        # (up to numerical precision)
        # Using higher tolerance for fp16 due to reduced precision
        torch.testing.assert_close(
            output_original, output_rotated, rtol=1e-2, atol=1e-2
        )


def test_model_equivariant_multiple_rotations_bf16(
    so3_group, single_point_cloud, model_bf16
):
    """Test equivariance with multiple random rotations using bf16 mixed precision."""
    # Run forward pass on original point cloud
    output_original = model_bf16(single_point_cloud)

    # Test multiple random rotations
    num_rotations = 5
    for _ in range(num_rotations):
        # Create a random rotation
        rotation = so3_group.sample()

        # Rotate the point cloud positions (x) and normals (n)
        rotated_pcd = single_point_cloud
        rotated_pcd.x = torch.matmul(rotation, single_point_cloud.x.T).T
        rotated_pcd.n = torch.matmul(rotation, single_point_cloud.n.T).T

        # Run forward pass on rotated point cloud
        output_rotated = model_bf16(rotated_pcd)

        # Check that outputs are on the same device
        assert output_original.device == output_rotated.device

        # Check that the outputs have the same shape
        assert output_original.shape == output_rotated.shape

        # Since the model should be equivariant, the outputs should be equal
        # (up to numerical precision)
        # Using higher tolerance for bf16 due to reduced precision
        torch.testing.assert_close(
            output_original, output_rotated, rtol=1e-2, atol=1e-2
        )
