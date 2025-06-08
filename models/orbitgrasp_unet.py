import math
import torch
import torch.nn as nn
from typing import Union

try:
    from e3nn import o3
except ImportError:
    pass

from equiformer_v2.gaussian_rbf import (
    GaussianRadialBasisLayer,
    GaussianRadialBasisLayerFiniteCutoff,
)
from equiformer_v2.edge_rot_mat import init_edge_rot_mat_deterministic
from equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2,
)
from equiformer_v2.module_list import ModuleListInfo
from equiformer_v2.radial_function import RadialFunction
from equiformer_v2.layer_norm import get_normalization_layer
from equiformerv2_block import (
    SO2EquivariantGraphAttention,
    TransBlock,
    FeedForwardNetwork,
)

from equiformer_v2.connectivity import RadiusGraph, FpsPool


class EquiformerUnet(nn.Module):

    def __init__(
        self,
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
        norm_type="rms_norm_sh",
        lmax_list=[3],
        mmax_list=[2],
        grid_resolution=None,
        edge_channels=(48, 48, 48),
        use_m_share_rad=False,
        distance_function="gaussian_soft",
        num_distance_basis=(256, 256, 256),
        use_attn_renorm=True,
        use_grid_mlp=True,
        use_sep_s2_act=True,
        alpha_drop=(0.1, 0.1, 0.1),
        drop_path_rate=(0.0, 0.0, 0.0),
        proj_drop=(0.1, 0.1, 0.1),
        weight_init="normal",
        mixed_precision: Union[bool, str] = None,  ### this can be fp16, fp32 or bf16
    ):
        super().__init__()

        self.max_neighbors = max_neighbors
        self.pool_ratio = pool_ratio
        self.cutoff = max_radius

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.edge_channels = edge_channels

        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.use_attn_renorm = use_attn_renorm
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]

        # Mixed precision setup
        self.use_mixed_precision = mixed_precision is not None
        if mixed_precision == "bf16":
            self.mixed_precision = torch.bfloat16
        elif mixed_precision == "fp16":
            self.mixed_precision = torch.float16
        elif mixed_precision is None:
            self.mixed_precision = torch.float32
        else:
            raise ValueError(f"Unknown mixed_precision: {mixed_precision}")

        self.max_radius = [max_radius[0]]

        for n, r in enumerate(max_radius[1:]):
            if r is None:
                self.max_radius.append(
                    self.max_radius[-1] / math.sqrt(self.pool_ratio[n - 1])
                )
            else:
                self.max_radius.append(r)

        self.deterministic = False
        self.device = torch.cuda.current_device()
        self.n_scales = len(self.max_radius)
        self.num_resolutions = len(self.lmax_list)
        self.pcd_channels = 3
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels[0]

        assert self.distance_function in ["gaussian", "gaussian_soft"]

        # Weights for message initialization
        self.sphere_linear_so3 = SO3_LinearV2(
            in_features=1,
            out_features=self.sphere_channels[0],
            lmax=max(self.lmax_list),
        )

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo(
            "({}, {})".format(max(self.lmax_list), max(self.lmax_list))
        )
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, m, resolution=self.grid_resolution, normalization="component"
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        ## Down Blocks
        self.down_blocks = torch.nn.ModuleList()
        for n in range(self.n_scales):
            # Initialize the sizes of radial functions (input channels and 2 hidden channels)
            edge_channels_list = [int(self.num_distance_basis[n])] + [
                self.edge_channels[n]
            ] * 2

            block = torch.nn.ModuleDict()
            block["pool"] = FpsPool(
                ratio=self.pool_ratio[n],
                random_start=not self.deterministic,
                r=self.max_radius[n],
                max_num_neighbors=self.max_neighbors[n],
            )
            block["radius_graph"] = RadiusGraph(
                r=self.max_radius[n], max_num_neighbors=1000
            )

            # Initialize the function used to measure the distances between atoms
            if self.distance_function == "gaussian":
                block["distance_expansion"] = GaussianRadialBasisLayer(
                    num_basis=self.num_distance_basis[n], cutoff=self.max_radius[n]
                )
            elif self.distance_function == "gaussian_soft":
                block["distance_expansion"] = GaussianRadialBasisLayerFiniteCutoff(
                    num_basis=self.num_distance_basis[n],
                    cutoff=self.max_radius[n] * 0.99,
                )
            else:
                raise ValueError

            block["transblock"] = TransBlock(
                sphere_channels=self.sphere_channels[n],
                attn_hidden_channels=self.attn_hidden_channels[n],
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels[n],
                attn_value_channels=self.attn_value_channels[n],
                ffn_hidden_channels=self.ffn_hidden_channels[n],
                output_channels=self.sphere_channels[n],
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                edge_channels_list=edge_channels_list,
                use_m_share_rad=self.use_m_share_rad,
                use_attn_renorm=self.use_attn_renorm,
                use_grid_mlp=self.use_grid_mlp,
                use_sep_s2_act=self.use_sep_s2_act,
                norm_type=self.norm_type,
                alpha_drop=self.alpha_drop[n],
                drop_path_rate=self.drop_path_rate[n],
                proj_drop=self.proj_drop[n],
            )
            layer_stack = torch.nn.ModuleList()
            if self.num_layers[n] > 1:
                if self.distance_function == "gaussian":
                    down_distance_expansion = GaussianRadialBasisLayer(
                        num_basis=self.num_distance_basis[n], cutoff=self.max_radius[n]
                    )
                elif self.distance_function == "gaussian_soft":
                    down_distance_expansion = GaussianRadialBasisLayerFiniteCutoff(
                        num_basis=self.num_distance_basis[n],
                        cutoff=self.max_radius[n] * 0.99,
                    )
                else:
                    raise ValueError
                layer_stack.append(down_distance_expansion)

                for _ in range(self.num_layers[n] - 1):
                    layer = torch.nn.ModuleDict()

                    layer["transblock"] = TransBlock(
                        sphere_channels=self.sphere_channels[n],
                        attn_hidden_channels=self.attn_hidden_channels[n],
                        num_heads=self.num_heads,
                        attn_alpha_channels=self.attn_alpha_channels[n],
                        attn_value_channels=self.attn_value_channels[n],
                        ffn_hidden_channels=self.ffn_hidden_channels[n],
                        output_channels=self.sphere_channels[n],
                        lmax_list=self.lmax_list,
                        mmax_list=self.mmax_list,
                        SO3_rotation=self.SO3_rotation,
                        mappingReduced=self.mappingReduced,
                        SO3_grid=self.SO3_grid,
                        edge_channels_list=edge_channels_list,
                        use_m_share_rad=self.use_m_share_rad,
                        use_attn_renorm=self.use_attn_renorm,
                        use_grid_mlp=self.use_grid_mlp,
                        use_sep_s2_act=self.use_sep_s2_act,
                        norm_type=self.norm_type,
                        alpha_drop=self.alpha_drop[n],
                        drop_path_rate=self.drop_path_rate[n],
                        proj_drop=self.proj_drop[n],
                    )
                    layer_stack.append(layer)
            block["layer_stack"] = layer_stack
            self.down_blocks.append(block)

        ## Middle Blocks
        self.middle_blocks = torch.nn.ModuleList()
        if self.distance_function == "gaussian":
            middle_distance_expansion = GaussianRadialBasisLayer(
                num_basis=self.num_distance_basis[-1], cutoff=self.max_radius[-1]
            )
        elif self.distance_function == "gaussian_soft":
            middle_distance_expansion = GaussianRadialBasisLayerFiniteCutoff(
                num_basis=self.num_distance_basis[-1], cutoff=self.max_radius[-1] * 0.99
            )
        else:
            raise ValueError
        self.middle_blocks.append(middle_distance_expansion)

        for i in range(self.num_layers[-1]):
            block = torch.nn.ModuleDict()

            # Initialize the sizes of radial functions (input channels and 2 hidden channels)
            edge_channels_list = [int(self.num_distance_basis[-1])] + [
                self.edge_channels[-1]
            ] * 2

            block["transblock"] = TransBlock(
                sphere_channels=self.sphere_channels[-1],
                attn_hidden_channels=self.attn_hidden_channels[-1],
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels[-1],
                attn_value_channels=self.attn_value_channels[-1],
                ffn_hidden_channels=self.ffn_hidden_channels[-1],
                output_channels=self.sphere_channels[-1],
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                edge_channels_list=edge_channels_list,
                use_m_share_rad=self.use_m_share_rad,
                use_attn_renorm=self.use_attn_renorm,
                use_grid_mlp=self.use_grid_mlp,
                use_sep_s2_act=self.use_sep_s2_act,
                norm_type=self.norm_type,
                alpha_drop=self.alpha_drop[-1],
                drop_path_rate=self.drop_path_rate[-1],
                proj_drop=self.proj_drop[-1],
            )
            self.middle_blocks.append(block)

        ## Up Blocks
        self.up_blocks = torch.nn.ModuleList()
        for n in range(self.n_scales - 1, -1, -1):
            block = torch.nn.ModuleDict()
            layer_stack = torch.nn.ModuleList()
            if self.num_layers[n] > 1:
                if self.distance_function == "gaussian":
                    up_distance_expansion = GaussianRadialBasisLayer(
                        num_basis=self.num_distance_basis[n], cutoff=self.max_radius[n]
                    )
                elif self.distance_function == "gaussian_soft":
                    up_distance_expansion = GaussianRadialBasisLayerFiniteCutoff(
                        num_basis=self.num_distance_basis[n],
                        cutoff=self.max_radius[n] * 0.99,
                    )
                else:
                    raise ValueError
                layer_stack.append(up_distance_expansion)

                for _ in range(self.num_layers[n] - 1):
                    layer = torch.nn.ModuleDict()

                    # Initialize the sizes of radial functions (input channels and 2 hidden channels)
                    edge_channels_list = [int(self.num_distance_basis[n])] + [
                        self.edge_channels[n]
                    ] * 2
                    layer["transblock"] = TransBlock(
                        sphere_channels=self.sphere_channels[n],
                        attn_hidden_channels=self.attn_hidden_channels[n],
                        num_heads=self.num_heads,
                        attn_alpha_channels=self.attn_alpha_channels[n],
                        attn_value_channels=self.attn_value_channels[n],
                        ffn_hidden_channels=self.ffn_hidden_channels[n],
                        output_channels=self.sphere_channels[n],
                        lmax_list=self.lmax_list,
                        mmax_list=self.mmax_list,
                        SO3_rotation=self.SO3_rotation,
                        mappingReduced=self.mappingReduced,
                        SO3_grid=self.SO3_grid,
                        edge_channels_list=edge_channels_list,
                        use_m_share_rad=self.use_m_share_rad,
                        use_attn_renorm=self.use_attn_renorm,
                        use_grid_mlp=self.use_grid_mlp,
                        use_sep_s2_act=self.use_sep_s2_act,
                        norm_type=self.norm_type,
                        alpha_drop=self.alpha_drop[n],
                        drop_path_rate=self.drop_path_rate[n],
                        proj_drop=self.proj_drop[n],
                    )
                    layer_stack.append(layer)
            block["layer_stack"] = layer_stack

            if n != 0:
                if self.distance_function == "gaussian":
                    block["distance_expansion"] = GaussianRadialBasisLayer(
                        num_basis=self.num_distance_basis[n], cutoff=self.max_radius[n]
                    )
                elif self.distance_function == "gaussian_soft":
                    block["distance_expansion"] = GaussianRadialBasisLayerFiniteCutoff(
                        num_basis=self.num_distance_basis[n],
                        cutoff=self.max_radius[n] * 0.99,
                    )
                else:
                    raise ValueError
                edge_channels_list = [int(self.num_distance_basis[n])] + [
                    self.edge_channels[n]
                ] * 2
                block["transblock"] = TransBlock(
                    sphere_channels=self.sphere_channels[n],
                    attn_hidden_channels=self.attn_hidden_channels[n],
                    num_heads=self.num_heads,
                    attn_alpha_channels=self.attn_alpha_channels[n],
                    attn_value_channels=self.attn_value_channels[n],
                    ffn_hidden_channels=self.ffn_hidden_channels[n],
                    output_channels=self.sphere_channels[n],
                    lmax_list=self.lmax_list,
                    mmax_list=self.mmax_list,
                    SO3_rotation=self.SO3_rotation,
                    mappingReduced=self.mappingReduced,
                    SO3_grid=self.SO3_grid,
                    edge_channels_list=edge_channels_list,
                    use_m_share_rad=self.use_m_share_rad,
                    use_attn_renorm=self.use_attn_renorm,
                    use_grid_mlp=self.use_grid_mlp,
                    use_sep_s2_act=self.use_sep_s2_act,
                    norm_type=self.norm_type,
                    alpha_drop=self.alpha_drop[n],
                    drop_path_rate=self.drop_path_rate[n],
                    proj_drop=self.proj_drop[n],
                )
            self.up_blocks.append(block)

        # Output blocks for point cloud features
        self.norm_1 = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels[0],
        )
        self.norm_2 = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels[0],
        )

        edge_channels_list = [int(self.num_distance_basis[0])] + [
            self.edge_channels[0]
        ] * 2

        self.output_layer = SO2EquivariantGraphAttention(
            sphere_channels=self.sphere_channels[0],
            hidden_channels=self.attn_hidden_channels[0],
            num_heads=self.num_heads,
            attn_alpha_channels=self.attn_alpha_channels[0],
            attn_value_channels=self.attn_value_channels[0],
            output_channels=1,
            lmax_list=self.lmax_list,
            mmax_list=self.mmax_list,
            SO3_rotation=self.SO3_rotation,
            mappingReduced=self.mappingReduced,
            SO3_grid=self.SO3_grid,
            edge_channels_list=edge_channels_list,
            use_m_share_rad=self.use_m_share_rad,
            use_attn_renorm=self.use_attn_renorm,
            use_sep_s2_act=self.use_sep_s2_act,
            alpha_drop=0.0,
        )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    def forward(self, pcds):

        if isinstance(pcds, list):
            batch_list = []
            node_coord_list = []
            node_feature_list = []
            batch_offset = 0
            num_points = 0
            for pcd in pcds:
                batch = pcd.b + batch_offset
                node_coord = pcd.x
                node_feature = pcd.n
                num_points += len(batch)

                batch_list.append(batch)
                node_coord_list.append(node_coord)
                node_feature_list.append(node_feature)
                batch_offset += 1

            batch = torch.cat(batch_list, dim=0)
            node_coord = torch.cat(node_coord_list, dim=0)
            node_feature = torch.cat(node_feature_list, dim=0)
        else:
            batch = pcds.b
            node_coord = pcds.x
            node_feature = pcds.n
            num_points = len(batch)

        self.dtype = node_coord.dtype
        self.device = node_coord.device

        node_src = None
        node_dst = None
        ########### Downstream Block #############
        downstream_outputs = []
        downstream_edges = []
        downstream_coords = []
        for n, block in enumerate(self.down_blocks):
            pool_graph = block["pool"](node_coord_src=node_coord, batch_src=batch)
            node_coord_dst, edge_src, edge_dst, degree, batch_dst, node_idx = pool_graph

            edge_distance_vec = node_coord.index_select(
                0, edge_src
            ) - node_coord_dst.index_select(0, edge_dst)
            edge_distance_vec = edge_distance_vec.detach()
            edge_distance = torch.norm(edge_distance_vec, dim=-1).detach()

            # Compute 3x3 rotation matrix per edge
            edge_rot_mat = self._init_edge_rot_mat(edge_distance_vec)

            # Initialize the WignerD matrices and other values for spherical harmonic calculations
            for i in range(self.num_resolutions):
                self.SO3_rotation[i].set_wigner(edge_rot_mat)

            if node_src is None:
                node_src = SO3_Embedding(
                    num_points,
                    self.lmax_list,
                    self.sphere_channels[n],
                    self.device,
                    self.dtype,
                )

                # offset_res = 0
                # offset = 0

                node_embedding = SO3_Embedding(
                    num_points,
                    self.lmax_list,
                    1,
                    self.device,
                    self.dtype,
                )
                node_embedding.embedding[:, 1:4, :] = node_feature.unsqueeze(-1)

                node_embedding = self.sphere_linear_so3(node_embedding)
                node_src.set_embedding(node_embedding.embedding)

            # Edge encoding (distance and atom edge)
            edge_attr = block["distance_expansion"](edge_distance)

            # Forward pass through model with mixed precision if CUDA is available
            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=self.use_mixed_precision,
                dtype=self.mixed_precision,
            ):
                node_dst = SO3_Embedding(
                    0,
                    self.lmax_list,
                    self.sphere_channels[n],
                    self.device,
                    self.dtype,
                )

            node_dst.set_embedding(node_src.embedding[node_idx])
            node_dst.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())
            node_dst = block["transblock"](
                node_src, node_dst, edge_attr, edge_src, edge_dst, batch=batch
            )
            node_src = node_dst
            node_coord_src = node_coord.clone()
            node_coord = node_coord_dst
            batch = batch_dst
            downstream_outputs.append((node_src, node_coord_src, node_coord_dst, batch))
            downstream_edges.append((edge_src, edge_dst, edge_distance, edge_rot_mat))
            downstream_coords.append((node_coord_src, node_coord_dst))

            if len(block["layer_stack"]) > 0:
                radiusGraph = block["radius_graph"](
                    node_coord_src=node_coord,
                    node_feature_src=node_dst,
                    batch_src=batch,
                )
                node_dst, node_coord_dst, edge_src, edge_dst, degree, batch_dst = (
                    radiusGraph
                )

                edge_distance_vec = node_coord.index_select(
                    0, edge_src
                ) - node_coord_dst.index_select(0, edge_dst)
                edge_distance_vec = edge_distance_vec.detach()
                edge_distance = torch.norm(edge_distance_vec, dim=-1).detach()

                # Compute 3x3 rotation matrix per edge
                edge_rot_mat = self._init_edge_rot_mat(edge_distance_vec)

                # Initialize the WignerD matrices and other values for spherical harmonic calculations
                for i in range(self.num_resolutions):
                    self.SO3_rotation[i].set_wigner(edge_rot_mat)

                edge_attr = block["layer_stack"][0](edge_distance)
                for n, layer in enumerate(block["layer_stack"][1:]):
                    node_dst = layer["transblock"](
                        node_src, node_dst, edge_attr, edge_src, edge_dst, batch=batch
                    )
                    node_src = node_dst
                    node_coord_src = node_coord.clone()
                    node_coord = node_coord_dst
                    batch = batch_dst
                    downstream_outputs.append(
                        (node_src, node_coord_src, node_coord_dst, batch)
                    )
                    downstream_edges.append(
                        (edge_src, edge_dst, edge_distance, edge_rot_mat)
                    )

        ########### Middle Block #############
        edge_attr = self.middle_blocks[0](edge_distance)
        for n, block in enumerate(self.middle_blocks[1:]):
            node_dst = block["transblock"](
                node_src, node_dst, edge_attr, edge_src, edge_dst, batch=batch
            )
            node_src = node_dst

        node_dst, node_coord_src, node_coord_dst, batch_dst = downstream_outputs.pop()
        node_src.embedding = (node_src.embedding + node_dst.embedding) / math.sqrt(
            3
        )  # Skip connection.

        ########### Upstream Block #############
        for n, block in enumerate(self.up_blocks):
            if len(block["layer_stack"]) != 0:
                edge_attr = None
                edge_rot_mat = None
                for i, layer in enumerate(block["layer_stack"][1:]):
                    node_dst, node_coord_src, node_coord_dst, batch_dst = (
                        downstream_outputs.pop()
                    )
                    edge_src, edge_dst, edge_distance, _ = downstream_edges.pop()
                    edge_src, edge_dst = (
                        edge_dst,
                        edge_src,
                    )  # Swap source and destination.
                    node_dst.embedding = (
                        node_src.embedding + node_dst.embedding
                    ) / math.sqrt(
                        3
                    )  # Skip connection.

                    edge_distance_vec = node_coord_dst.index_select(
                        0, edge_src
                    ) - node_coord_dst.index_select(0, edge_dst)
                    edge_distance_vec = edge_distance_vec.detach()
                    edge_distance = torch.norm(edge_distance_vec, dim=-1).detach()

                    if edge_attr is None:
                        edge_attr = block["layer_stack"][0](edge_distance)
                    else:
                        edge_attr = edge_attr
                    if edge_rot_mat is None:
                        # Compute 3x3 rotation matrix per edge
                        edge_rot_mat = self._init_edge_rot_mat(edge_distance_vec)
                        for ii in range(self.num_resolutions):
                            self.SO3_rotation[ii].set_wigner(edge_rot_mat)
                    else:
                        edge_rot_mat = edge_rot_mat

                    # Initialize the WignerD matrices and other values for spherical harmonic calculations

                    node_dst = block["layer_stack"][i + 1]["transblock"](
                        node_src,
                        node_dst,
                        edge_attr,
                        edge_src,
                        edge_dst,
                        batch=batch_dst,
                    )
                    node_src = node_dst
                    batch = batch_dst

            if n != self.n_scales - 1:
                node_dst, node_coord_src, node_coord_dst, batch_dst = (
                    downstream_outputs.pop()
                )
                node_coord_src, node_coord_dst = downstream_coords.pop()
                edge_src, edge_dst, edge_distance, _ = downstream_edges.pop()
                edge_src, edge_dst = edge_dst, edge_src

                edge_distance_vec = node_coord_dst.index_select(
                    0, edge_src
                ) - node_coord_src.index_select(0, edge_dst)
                edge_distance_vec = edge_distance_vec.detach()
                edge_distance = torch.norm(edge_distance_vec, dim=-1).detach()

                # Compute 3x3 rotation matrix per edge
                edge_rot_mat = self._init_edge_rot_mat(edge_distance_vec)
                # Initialize the WignerD matrices and other values for spherical harmonic calculations
                for i in range(self.num_resolutions):
                    self.SO3_rotation[i].set_wigner(edge_rot_mat)
                edge_attr = block["distance_expansion"](edge_distance)
                node_dst = block["transblock"](
                    node_src, node_dst, edge_attr, edge_src, edge_dst, batch=batch
                )
                node_src = node_dst
                batch = batch_dst

        # Final layer norm
        node_src.embedding = self.norm_1(node_src.embedding)
        node_dst.embedding = self.norm_2(node_dst.embedding)
        point_features = self.output_layer(
            node_src, node_dst, edge_attr, edge_src, edge_dst
        )

        return point_features

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, edge_distance_vec):
        # return init_edge_rot_mat(edge_distance_vec)
        return init_edge_rot_mat_deterministic(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, SO3_LinearV2):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)
