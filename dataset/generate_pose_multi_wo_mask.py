import numpy as np
import sys
from pathlib import Path
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import open3d as o3d
from torch_geometric.nn import radius
import torch
import pickle
from tqdm import trange
import os
import argparse
from simulator.simulation_clutter_bandit_multi_camera import ClutterRemovalSim
from simulator.camera import RealSenseD415
from utils.transform import Transform, Rotation
from utils.utils_3d import AngleBasedSampler, FarthestSampler
from data_generation_utils import (
    show_pcds_gt_segs,
    show_grasp_indices,
    get_gripper_angle_mask,
    voxel_down_sample,
)


def data_save(
    filename,
    mode,
    pcd_points,
    pcd_normals,
    sample_index,
    grasping_pose,
    success,
    object_num,
):
    assert mode in ["ab", "wb"]

    local_vars = locals()
    data_to_save = {
        k: local_vars[k]
        for k in [
            "pcd_points",
            "pcd_normals",
            "sample_index",
            "grasping_pose",
            "success",
            "object_num",
        ]
    }

    with open(filename, mode) as f:
        pickle.dump(data_to_save, f)  # type: ignore


def run(scene, object_set, iteration_num, savePath, start=0, GUI=True, n_intervals=36):
    sim = ClutterRemovalSim(
        scene,
        object_set,
        gui=GUI,
        rand=True,
        seed=start,
        load_sam=False,
        n_intervals=n_intervals,
    )

    total_grasp = 0
    total_success = 0

    # Because we don't have segmentation masks, we need to generate the sub-point clouds by FPS
    point_sampler = FarthestSampler()
    sample_strategy = AngleBasedSampler()

    object_index = None
    position = None
    color = None

    for itr in trange(iteration_num, desc="Iteration"):

        objs_num = sim.rng.poisson(4) + 1
        sim.reset(
            objs_num, index=object_index, pose_=position, color_=color, from_save=False
        )

        seg_gts = sim.segmaps
        if sim.num_objects == 0:
            continue
        sim.save_state()

        length = sim.pcds_xyz.shape[0] // 3

        pcds_all = list()
        index_list = list()
        # Generate the point cloud for each camera
        for j, colormap in enumerate(sim.colormaps):
            start_idx = j * length
            end_idx = (j + 1) * length
            pcd_xyz = sim.pcds_xyz[start_idx:end_idx][
                sim.non_zero_indices[start_idx:end_idx]
            ]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_xyz)

            if args.add_noise:
                vertices = np.asarray(pcd.points)
                # add gaussian noise 95% confident interval (-1.96,1.96)
                vertices = vertices + np.random.normal(
                    loc=0.0, scale=0.0005, size=(len(vertices), 3)
                )
                pcd.points = o3d.utility.Vector3dVector(vertices)

            pcd, ind_1 = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2)
            pcd, ind_2 = pcd.remove_radius_outlier(nb_points=30, radius=0.03)

            original_normals = np.zeros((len(pcd.points), 3), dtype=np.float64)

            pcd.normals = o3d.utility.Vector3dVector(original_normals)

            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.05, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(30)
            pcd.orient_normals_towards_camera_location(
                camera_location=RealSenseD415.CONFIG[j]["position"]
            )
            pcds_all.append(pcd)
            index_list.append([ind_1, ind_2])

        # Generate the combined point cloud
        all_pcds_gt_segs = []
        for i in range(len(sim.colormaps)):
            pcds_gt_segs = seg_gts[i].reshape(-1, 1)[
                sim.non_zero_indices[i * length : (i + 1) * length]
            ]
            pcds_gt_segs = pcds_gt_segs[index_list[i][0]]
            pcds_gt_segs = pcds_gt_segs[index_list[i][1]].squeeze(-1)
            all_pcds_gt_segs.append(pcds_gt_segs)

        all_points = np.vstack(
            (
                np.asarray(pcds_all[0].points),
                np.asarray(pcds_all[1].points),
                np.asarray(pcds_all[2].points),
            )
        )

        all_pcds_gt_segs = np.concatenate(
            [all_pcds_gt_segs[i] for i in range(len(all_pcds_gt_segs))]
        )
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        all_normals = np.vstack(
            (
                np.asarray(pcds_all[0].normals),
                np.asarray(pcds_all[1].normals),
                np.asarray(pcds_all[2].normals),
            )
        )
        combined_pcd.normals = o3d.utility.Vector3dVector(all_normals)

        # Downsample the point cloud
        down_pcd, down_gt_segs = voxel_down_sample(
            combined_pcd, voxel_size=0.0055, gt_seg_arr=all_pcds_gt_segs
        )

        if SHOW_PCD_MASK:
            show_pcds_gt_segs(down_pcd, down_gt_segs)

        action_all_masks = list()
        success_list_all = list()
        down_pcd_points = np.asarray(down_pcd.points)
        down_pcd_normals = np.asarray(down_pcd.normals)

        grasp_indices = list()

        # Sample the center points
        sample_center_points, sample_center_index = point_sampler(
            down_pcd_points, sim.num_objects * 2, 0.055
        )
        down_pcd_points_tensor = torch.tensor(
            down_pcd_points, dtype=torch.float32, requires_grad=False
        )

        # Generate the grasping poses
        for order, center_point in enumerate(sample_center_points):
            center_point = torch.tensor(
                center_point, dtype=torch.float32, requires_grad=False
            ).reshape(1, 3)
            edge_index = radius(
                down_pcd_points_tensor, center_point, r=0.04, max_num_neighbors=900
            )
            points_down_mask = down_pcd_points[edge_index[1]]
            normals_down_mask = down_pcd_normals[edge_index[1]]

            local_pcd = o3d.geometry.PointCloud()
            local_pcd.points = o3d.utility.Vector3dVector(points_down_mask)
            local_pcd.normals = o3d.utility.Vector3dVector(normals_down_mask)

            actions_list, points_list, normals_list, indices_list = (
                sim.mask_random_actions(
                    local_pcd,
                    sample_strategy,
                    m=sample_num,
                    n_intervals=n_intervals,
                    z_thresh=0.052,
                )
            )

            global_indices = edge_index[1][indices_list]
            grasp_indices.append(global_indices)

            action_info = {
                "actions": actions_list,
                "points": points_list,
                "normals": normals_list,
                "indices": indices_list,
                "grasp_indices": global_indices,
                "order": order,
            }
            action_all_masks.append(action_info)

        if SHOW_GRASP_POINTS:
            show_grasp_indices(down_pcd, grasp_indices, random_color=False)

        succ_flag = 0
        total_flag = 0

        pbar = trange(len(action_all_masks), desc="Actions", leave=False)
        # Execute the grasping actions and evaluate the success
        # Some geometry priors are used to filter out the invalid grasping poses
        for p_index in pbar:
            actions = action_all_masks[p_index]["actions"]
            normals = action_all_masks[p_index]["normals"]
            grasp_points = grasp_indices[p_index]
            success_list = list()
            for i, normal in enumerate(normals):
                grasp_point = grasp_points[i]

                for action in actions[i]:
                    total_grasp += 1
                    pos = action[:3]
                    rot = action[-4:]

                    gripper_angle_mask = get_gripper_angle_mask(
                        Rotation.from_quat(rot), threshold=75
                    )
                    if gripper_angle_mask:
                        result, grasped_object_ind, graspable = sim.execute_grasp(
                            action=[pos, rot, normal], waitTime=0, remove=False
                        )

                        if graspable == 0:
                            success_list.append(-1)
                        else:
                            success, object_id = sim.eval_single_grasp(
                                index=grasp_point,
                                segmentation_gt=down_gt_segs,
                                grasp_obj_id=grasped_object_ind,
                            )
                            if success:
                                succ_flag += 1
                                total_success += 1
                                success_list.append(1)
                            else:
                                success_list.append(0)
                    else:
                        success_list.append(-1)
                    total_flag += 1
                    pbar.set_postfix(
                        {
                            "sratio:": f"{succ_flag}/ {total_flag}",
                            "total sratio:": f"{total_success}/ {total_grasp}",
                        }
                    )

                    sim.restore_state()
            success_list = np.asarray(success_list).reshape(
                len(normals), n_intervals, 1
            )
            success_list_all.append(success_list)

        # Save the data
        if SAVING:
            save_dir = os.path.join(savePath, f"se3_origin/{scene}_multi")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = os.path.join(save_dir, f"{scene}_{(start + itr):05}.pkl")
            mode = "wb"

            data_save(
                filename=filename,
                mode=mode,
                pcd_points=down_pcd_points,
                pcd_normals=down_pcd_normals,
                sample_index=sample_center_index,
                grasping_pose=action_all_masks,
                success=success_list_all,
                object_num=sim.num_objects,
            )

    sim.close()


if __name__ == "__main__":
    sys.setrecursionlimit(150000)
    np.set_printoptions(threshold=np.inf)

    def strToBool(value):
        if value.lower() in {"false", "f", "0", "no", "n"}:
            return False
        elif value.lower() in {"true", "t", "1", "yes", "y"}:
            return True
        raise ValueError(f"{value} is not a valid boolean value")

    parser = argparse.ArgumentParser()
    # Visualization
    parser.add_argument("--GUI", type=strToBool, default=False)
    parser.add_argument("--show_grasp_points", type=strToBool, default=False)
    parser.add_argument("--show_pcds_mask", type=strToBool, default=False)
    # Saving
    parser.add_argument("--save_data", type=strToBool, default=True)
    parser.add_argument(
        "--save_path",
        type=str,
        default=Path(__file__).resolve().parent / "collected_data",
    )
    # Data generation configurations
    parser.add_argument("--n_intervals", type=int, default=36)
    parser.add_argument("--start_scene", type=int, default=0)
    parser.add_argument("--iteration_num", type=int, default=50)
    parser.add_argument("--FilterSame", type=strToBool, default=True)
    parser.add_argument("--sample_num", type=int, default=20)
    parser.add_argument("--add_noise", type=strToBool, default=True)
    parser.add_argument("--object_set", type=str, default="train")

    parser.add_argument("--scene", type=str, default="packed")

    args = parser.parse_args()
    SAVING = args.save_data
    GUI = args.GUI
    sample_num = args.sample_num
    n_intervals = args.n_intervals
    SHOW_GRASP_POINTS = args.show_grasp_points
    SHOW_PCD_MASK = args.show_pcds_mask
    PATH = args.save_path
    START = args.start_scene
    ITERATIONNUM = args.iteration_num
    FilterSame = args.FilterSame

    scene = args.scene
    object_set = args.object_set

    run(
        scene,
        object_set,
        ITERATIONNUM,
        PATH,
        start=START,
        GUI=GUI,
        n_intervals=n_intervals,
    )
