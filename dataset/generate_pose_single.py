import numpy as np
import sys
from pathlib import Path
import os
from math import sin, cos

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import open3d as o3d
import pickle
from tqdm import trange
import argparse
from simulator.simulation_clutter_bandit_single_camera import ClutterRemovalSim
from simulator.sam3d import filter_masks
from utils.transform import Transform, Rotation
from utils.utils_3d import AngleBasedSampler
from data_generation_utils import (
    get_gripper_angle_mask,
    show_pcds_masks,
    show_pcds_gt_segs,
    voxel_down_sample_with_multiple_masks,
)


def data_save(
    filename,
    mode,
    pcd_points,
    pcd_normals,
    all_masks,
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
            "all_masks",
            "grasping_pose",
            "success",
            "object_num",
        ]
    }

    with open(filename, mode) as f:
        pickle.dump(data_to_save, f)  # type: ignore


def run(
    scene,
    object_set,
    iteration_num,
    savePath,
    from_save=True,
    start=0,
    GUI=True,
    n_intervals=36,
):
    sim = ClutterRemovalSim(
        scene, object_set, gui=GUI, rand=True, seed=12345, n_intervals=n_intervals
    )
    masks_ori = None

    total_grasp = 0
    total_success = 0

    if not from_save:
        sim.loadSAM()

    # sample_strategy = FarthestSampler()
    sample_strategy = AngleBasedSampler()

    object_index = None
    position = None
    color = None
    camera_config = None

    for itr in trange(iteration_num, desc="Iteration"):
        if from_save:
            with open(
                Path(__file__).resolve().parent
                / f"store/RawData_{scene}_single/Indexes/index_{(itr + start):05}.pkl",
                "rb",
            ) as f:
                while True:
                    try:
                        object_index = pickle.load(f)
                    except EOFError:  # End of File
                        break

            with open(
                Path(__file__).resolve().parent
                / f"store/RawData_{scene}_single/Poses/pose_{(itr + start):05}.pkl",
                "rb",
            ) as f:
                while True:
                    try:
                        position = pickle.load(f)
                    except EOFError:  # End of File
                        break

            with open(
                Path(__file__).resolve().parent
                / f"store/RawData_{scene}_single/Masks/mask_{(itr + start):05}.pkl",
                "rb",
            ) as f:
                while True:
                    try:
                        masks_ori = pickle.load(f)
                    except EOFError:  # End of File
                        break

            with open(
                Path(__file__).resolve().parent
                / f"store/RawData_{scene}_single/Colors/color_{(itr + start):05}.pkl",
                "rb",
            ) as f:
                while True:
                    try:
                        color = pickle.load(f)
                    except EOFError:  # End of File
                        break
            with open(
                Path(__file__).resolve().parent
                / f"store/RawData_{scene}_single/Camera/camera_{(itr + start):05}.pkl",
                "rb",
            ) as f:
                while True:
                    try:
                        camera_config = pickle.load(f)
                    except EOFError:  # End of File
                        break

            objs_num = -1

        else:
            objs_num = sim.rng.poisson(4) + 1
            r = np.random.uniform(1.5, 2) * sim.size
            theta = np.random.uniform(np.pi / 4, np.pi / 2.4)
            phi = np.random.uniform(0.0, np.pi)
            origin = Transform(
                Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0 + 0.15]
            )
            eye = np.r_[
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta),
            ]
            eye = eye + origin.translation

            a = (
                Transform.look_at(
                    eye - origin.translation, np.array([0, 0, 0]), [0, 0, 1]
                )
                * origin.inverse()
            )
            a = a.inverse().to_list()
            camera_config = {
                "image_size": (480, 640),
                "intrinsics": (450.0, 0, 320.0, 0, 450.0, 240.0, 0, 0, 1),
                "position": eye,
                "rotation": a[:4],
                "zrange": (0.01, 2.0),
                "noise": False,
                "name": "random",
            }
            #######################################################################################
            # original camera setting of edge grasp.
            # ours is slightly different with their setting.
            # Meanwhile, the camera intrinsic is different, so if you need to generate grasp pose
            # on the original setting, please use the following setting and camera intrinsic.
            # NOTE: remember to change the camera intrinsic in the generate_raw_pose_mask_single.py

            # r = np.random.uniform(2, 2.5) * sim.size
            # theta = np.random.uniform(np.pi / 4, np.pi / 3)
            # phi = np.random.uniform(0.0, np.pi)
            # origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0 + 0.25])
            # eye = np.r_[
            #     r * sin(theta) * cos(phi),
            #     r * sin(theta) * sin(phi),
            #     r * cos(theta),
            # ]
            # eye = eye + origin.translation
            #
            # a = Transform.look_at(eye - origin.translation, np.array([0, 0, 0]), [0, 0, 1]) * origin.inverse()
            # a = a.inverse().to_list()
            # config = {
            #     'image_size': (480, 640),
            #     'intrinsics': (540., 0, 320., 0, 540., 240., 0, 0, 1),   ### original camera intrinsic
            #     'position': eye,
            #     'rotation': a[:4],
            #     'zrange': (0.01, 2.),
            #     'noise': False,
            #     'name': 'random'
            # }
            #######################################################################################

        sim.reset(
            objs_num,
            index=object_index,
            pose_=position,
            color_=color,
            from_save=from_save,
        )
        sim.getRandomViewObservation(camera_config)
        img = np.load(
            Path(__file__).resolve().parent
            / f"store/RawData_{scene}_single/Images/rgb_{itr + start:05}.npy"
        )
        img_depth = np.load(
            Path(__file__).resolve().parent
            / f"store/RawData_{scene}_single/Images/depth_{itr + start:05}.npy"
        )
        print("Equal Depth Value:", np.array_equal(sim.heightmaps, img_depth))
        print("Equal RGB Value:", np.array_equal(sim.colormaps, img))
        seg_gts = sim.segmaps
        if sim.num_objects == 0:
            continue
        sim.save_state()

        if not from_save:
            sim.get_segmentation()
            masks_ori = sim.maskmaps
            masks_ori = masks_ori[0]

        masks_filtered = list()
        pcds_all = list()
        index_list = list()
        masks = filter_masks(
            masks_ori, sim.pcds_xyz, 0.3, bound_ratio=sim.bound_radio, offset=15
        )
        masks_filtered.append(masks)
        pcd_xyz = sim.pcds_xyz[sim.non_zero_indices]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_xyz)

        if args.add_noise:
            vertices = np.asarray(pcd.points)
            # add gaussian noise 95% confident interval (-1.96,1.96)
            vertices = vertices + np.random.normal(
                loc=0.0, scale=0.0008, size=(len(vertices), 3)
            )
            pcd.points = o3d.utility.Vector3dVector(vertices)

        pcd, ind_1 = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=3)
        pcd, ind_2 = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(30)
        pcd.orient_normals_towards_camera_location(
            camera_location=camera_config["position"]
        )
        pcds_all.append(pcd)
        index_list.append([ind_1, ind_2])

        pcds_masks = []
        if len(masks) != 0:
            for mask in masks:
                pcds_masks.append(mask["segmentation"])
        else:
            continue

        pcds_masks = np.asarray(pcds_masks).reshape(len(pcds_masks), -1, 1)[
            :, sim.non_zero_indices
        ]

        noise_indices = sim.pcds_xyz[:, 2][sim.non_zero_indices] < 0.052
        pcds_masks[:, noise_indices] = False
        pcds_gt_segs = seg_gts.reshape(-1, 1)[sim.non_zero_indices]
        pcds_gt_segs = pcds_gt_segs[index_list[0][0]]
        pcds_gt_segs = pcds_gt_segs[index_list[0][1]].squeeze(-1)

        pcds_masks = pcds_masks[:, index_list[0][0]]
        pcds_masks = pcds_masks[:, index_list[0][1]].squeeze(-1)

        all_points = np.asarray(pcd.points)
        all_normals = np.asarray(pcd.normals)

        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        combined_pcd.normals = o3d.utility.Vector3dVector(all_normals)

        down_pcd, down_mask_list, down_gt_segs = voxel_down_sample_with_multiple_masks(
            combined_pcd,
            voxel_size=0.0055,
            masks_arr=pcds_masks,
            gt_seg_arr=pcds_gt_segs,
        )

        points_array = np.asarray(down_pcd.points)
        filtered_down_mask_list = [
            downMask
            for downMask in down_mask_list
            if downMask.sum() > 0 and points_array[downMask][:, -1].mean() > 0.053
        ]
        down_mask_list = filtered_down_mask_list

        if SHOW_PCD_MASK:
            show_pcds_masks(down_pcd, down_mask_list)
            show_pcds_gt_segs(down_pcd, down_gt_segs)

        action_all_masks = list()
        success_list_all = list()
        down_pcd_points = np.asarray(down_pcd.points)
        down_pcd_normals = np.asarray(down_pcd.normals)

        grasp_indices = list()

        for order, down_mask in enumerate(down_mask_list):
            if down_mask.sum() == 0:
                continue
            points_down_mask = down_pcd_points[down_mask]
            normals_down_mask = down_pcd_normals[down_mask]

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

            select_idx = np.where(down_mask == True)[0]
            grasp_indices.append(select_idx[indices_list])

            action_info = {
                "actions": actions_list,
                "points": points_list,
                "normals": normals_list,
                "indices": indices_list,
                "grasp_indices": select_idx[indices_list],
                "order": order,
            }
            action_all_masks.append(action_info)

        if SHOW_GRASP_POINTS:
            show_pcds_masks(down_pcd, grasp_indices, random_color=False)

        succ_flag = 0
        total_flag = 0

        pbar = trange(len(action_all_masks), desc="Actions", leave=False)
        for p_index in pbar:
            actions = action_all_masks[p_index]["actions"]
            normals = action_all_masks[p_index]["normals"]
            grasp_points = grasp_indices[p_index]
            success_list = list()
            for i, normal in enumerate(normals):
                grasp_point = grasp_points[i]
                # normal_mask = check_normal_availability(normal)

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
        if SAVING:
            save_dir = os.path.join(savePath, f"se3_origin/{scene}_single")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = os.path.join(save_dir, f"{scene}_{(start + itr):05}.pkl")
            mode = "wb"

            data_save(
                filename=filename,
                mode=mode,
                pcd_points=down_pcd_points,
                pcd_normals=down_pcd_normals,
                all_masks=down_mask_list,
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
    parser.add_argument("--scene", type=str, default="pile")
    parser.add_argument("--from_save", type=strToBool, default=True)

    parser.add_argument("--save_data", type=strToBool, default=True)
    parser.add_argument("--GUI", type=strToBool, default=False)
    parser.add_argument("--show_grasp_pcds", type=strToBool, default=False)
    parser.add_argument("--random_min", type=int, default=10)
    parser.add_argument("--random_max", type=int, default=15)

    parser.add_argument("--n_intervals", type=int, default=36)
    parser.add_argument("--show_pcds", type=strToBool, default=False)
    parser.add_argument(
        "--save_path",
        type=str,
        default=Path(__file__).resolve().parent / "collected_data",
    )
    parser.add_argument("--start_scene", type=int, default=0)
    parser.add_argument("--iteration_num", type=int, default=40)
    parser.add_argument("--FilterSame", type=strToBool, default=True)
    parser.add_argument("--sample_num", type=int, default=20)
    parser.add_argument("--add_noise", type=strToBool, default=True)

    args = parser.parse_args()

    scene = args.scene
    SAVING = args.save_data
    GUI = args.GUI
    LOW, HIGH = args.random_min, args.random_max
    sample_num = args.sample_num
    n_intervals = args.n_intervals
    SHOW_GRASP_POINTS = args.show_grasp_pcds
    SHOW_PCD_MASK = args.show_pcds
    PATH = args.save_path
    START = args.start_scene
    ITERATIONNUM = args.iteration_num
    FilterSame = args.FilterSame

    object_set = "train"

    run(
        scene,
        object_set,
        ITERATIONNUM,
        PATH,
        from_save=args.from_save,
        start=START,
        GUI=GUI,
        n_intervals=n_intervals,
    )
