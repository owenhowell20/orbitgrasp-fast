import sys
import numpy as np
import os
import torch
import open3d as o3d
import tqdm
import yaml
from math import cos, sin
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from se3_grasper_bce import OrbitGrasper
from utils.torch_utils import set_seed, write_test, write_log
from simulator.simulation_clutter_bandit_single_camera import ClutterRemovalSim
from utils.FeaturedPoints import FeaturedPoints
from e3nn.o3 import spherical_harmonics_alpha_beta
from utils.transform import Rotation, Transform
from utils.utils_3d import FarthestSampler
from collections import Counter
from torch_geometric.nn import radius
from utils.utility import get_gripper_points_mask


def grasp_test_random(
    initial_radius=0.035,
    max_allowed_points=800,
    min_allowed_points=700,
    min_radius=-1e-6,
    max_radius=0.15,
    lmax=3,
    config_file=None,
):
    base_path = Path(__file__).resolve().parent
    load_name = find_checkpoint(
        base_path / config_file["test"]["root_dir"], config_file["test"]["load_epoch"]
    )
    orbit_grasper = OrbitGrasper(
        device=config_file["orbit_grasper"]["device"],
        load=config_file["test"]["load_epoch"],
        param_dir=base_path / config_file["test"]["root_dir"],
        num_channel=config_file["orbit_grasper"]["num_channel"],
        lmax=3,
        mmax=2,
        load_name=load_name,
        training_config=config_file,
    )

    scene = config_file["test"]["scene"]
    object_set = "test"
    GUI = config_file["test"]["GUI"]

    record = list()
    success_list = list()
    declutter_list = list()
    point_sampler = FarthestSampler()

    for RUN in range(config_file["test"]["RUN_TIMES"]):
        success_rate = 0
        declutter_rate = 0
        num_rounds = config_file["test"]["NUM_ROUNDS"]
        silence = False
        cnt = 0
        success = 0
        total_objs = 0
        grasped_objs = 0
        remain_objs = 0
        set_seed(RUN + config_file["seed"])

        sim = ClutterRemovalSim(
            scene, object_set, rand=True, gui=GUI, seed=RUN + config_file["seed"]
        )

        # control the gui speed
        sim.world.sleep = 0.001

        for _ in tqdm.tqdm(range(num_rounds), disable=silence):
            objs_num = 5

            sim.reset(objs_num)
            total_objs += sim.num_objects
            consecutive_failures = 1
            last_label = True
            skip_time = 0

            while sim.num_objects > 0 and consecutive_failures < 2 and skip_time < 3:

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
                config = {
                    "image_size": (480, 640),
                    "intrinsics": (450.0, 0, 320.0, 0, 450.0, 240.0, 0, 0, 1),
                    "position": eye,
                    "rotation": a[:4],
                    "zrange": (0.01, 2.0),
                    "noise": False,
                    "name": "random",
                }

                # original camera setting of edge grasp
                # ours is slightly different with their setting. Meanwhile, the camera intrinsic is different, so if you need to test our method
                # on the original setting, please use the following setting and camera intrinsic to collect data and train the network.
                # NOTE: you can't directly change the camera intrinsic to test on different envs, you need to retrain the network.

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
                #     'intrinsics': (540., 0, 320., 0, 540., 240., 0, 0, 1),
                #     'position': eye,
                #     'rotation': a[:4],
                #     'zrange': (0.01, 2.),
                #     'noise': False,
                #     'name': 'random'
                # }
                sim.getRandomViewObservation(config)

                pcd_xyz = sim.pcds_xyz[sim.non_zero_indices]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
                if config_file["test"]["add_noise"]:
                    vertices = np.asarray(pcd.points)
                    # add gaussian noise 95% confident interval (-1.96,1.96)
                    vertices = vertices + np.random.normal(
                        loc=0.0, scale=0.0005, size=(len(vertices), 3)
                    )
                    pcd.points = o3d.utility.Vector3dVector(vertices)
                pcd, ind_1 = pcd.remove_statistical_outlier(
                    nb_neighbors=30, std_ratio=4.0
                )
                pcd, ind_2 = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.04, max_nn=30
                    )
                )
                pcd.orient_normals_consistent_tangent_plane(30)
                pcd.orient_normals_towards_camera_location(camera_location=eye)

                # all_points = np.asarray(pcd.points)
                # all_normals = np.asarray(pcd.normals)
                #
                # combined_pcd = o3d.geometry.PointCloud()
                # combined_pcd.points = o3d.utility.Vector3dVector(all_points)
                # combined_pcd.normals = o3d.utility.Vector3dVector(all_normals)

                down_pcd = voxel_down_sample(pcd, voxel_size=0.0055)
                # o3d.visualization.draw_geometries([down_pcd])

                down_pcd_points = (
                    torch.from_numpy(np.asarray(down_pcd.points))
                    .float()
                    .to(config_file["orbit_grasper"]["device"])
                    .detach()
                )
                down_pcd_normals = (
                    torch.from_numpy(np.asarray(down_pcd.normals))
                    .float()
                    .to(config_file["orbit_grasper"]["device"])
                    .detach()
                )

                data_list = list()
                grasping_indices_list = list()
                sphere_indices_list = list()
                harmonics_list = list()
                grasp_poses_list = list()
                sample_center_points, sample_center_index = point_sampler(
                    down_pcd_points.cpu().numpy(), 10, 0.052
                )
                down_pcd_points_tensor = torch.tensor(
                    down_pcd_points, dtype=torch.float32, requires_grad=False
                )

                for order, center_point in enumerate(sample_center_points):
                    center_point = (
                        torch.tensor(
                            center_point, dtype=torch.float32, requires_grad=False
                        )
                        .reshape(1, 3)
                        .to(config_file["orbit_grasper"]["device"])
                    )
                    edge_index = radius(
                        down_pcd_points_tensor,
                        center_point,
                        r=0.05,
                        max_num_neighbors=900,
                    )
                    points_down_mask = down_pcd_points[edge_index[1]]
                    normals_down_mask = down_pcd_normals[edge_index[1]]

                    grasp_poses, points_list, normals_list, indices_list = (
                        sim.mask_all_actions_batch(
                            points_down_mask,
                            normals_down_mask,
                            n_intervals=config_file["orbit_grasper"]["num_channel"],
                            z_thresh=0.052,
                        )
                    )

                    if indices_list is None:
                        continue
                    grasp_indices = edge_index[1][indices_list]

                    local_points = down_pcd_points_tensor[edge_index[1]]
                    if local_points.shape[0] == 0:
                        continue
                    mask_center = local_points.mean(dim=0)
                    distances = torch.norm(local_points - mask_center, dim=1)

                    # Find all points within this expanded radius from the centroid
                    temp_radius = initial_radius
                    itr = 0
                    while True:
                        mask_radius = distances.max() + temp_radius
                        all_distances_to_center = torch.norm(
                            down_pcd_points - mask_center, dim=1
                        )
                        sphere_mask = all_distances_to_center <= mask_radius
                        sphere_points = down_pcd_points[sphere_mask]
                        sphere_normals = down_pcd_normals[sphere_mask]

                        if (
                            sphere_points.shape[0] <= max_allowed_points
                            and sphere_points.shape[0] >= min_allowed_points
                        ):
                            break
                        elif sphere_points.shape[0] > max_allowed_points:
                            temp_radius -= 0.005
                            if temp_radius <= min_radius:
                                break
                        elif sphere_points.shape[0] < min_allowed_points:
                            temp_radius += 0.005
                            if temp_radius > max_radius:
                                break
                        itr += 1
                        if itr >= 30:
                            break

                    if sphere_points.shape[0] > (
                        max_allowed_points + 100
                    ) or sphere_points.shape[0] < (min_allowed_points - 100):
                        continue
                    #     visulaize sphere points
                    #     pcd = o3d.geometry.PointCloud()
                    #     pcd.points = o3d.utility.Vector3dVector(sphere_points.cpu().numpy())
                    #     o3d.visualization.draw_geometries([pcd])
                    #  visulaize original masked points
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(points_down_mask.cpu().numpy())
                    # o3d.visualization.draw_geometries([pcd])
                    grasp_indices = grasp_indices.to(
                        config_file["orbit_grasper"]["device"]
                    )
                    grasp_quats = grasp_poses[:, :, -4:]
                    grasp_z = quaternion_to_z_direction(grasp_quats)
                    mask_harmonics = compute_spherical_harmonics(grasp_z, lmax=lmax)

                    local_indices = torch.arange(
                        down_pcd_points.shape[0],
                        device=config_file["orbit_grasper"]["device"],
                    )[sphere_mask]
                    # get the local indices in the sphere points
                    sphere_indices, reorderded_indices = torch.where(
                        local_indices.unsqueeze(1) == grasp_indices.unsqueeze(0)
                    )

                    reordered_grasp_poses = grasp_poses[reorderded_indices]
                    reordered_mask_harmonics = mask_harmonics[reorderded_indices.cpu()]

                    feature_points = FeaturedPoints(
                        x=sphere_points,
                        n=sphere_normals,
                        b=torch.ones(
                            sphere_points.shape[0],
                            dtype=torch.long,
                            device=config_file["orbit_grasper"]["device"],
                        ),
                    )

                    feature_points = normalize(feature_points)
                    data_list.append(feature_points)
                    sphere_indices_list.append(sphere_indices)
                    grasping_indices_list.append(grasp_indices)
                    harmonics_list.append(
                        reordered_mask_harmonics.to(
                            config_file["orbit_grasper"]["device"]
                        )
                    )
                    grasp_poses_list.append(reordered_grasp_poses)

                if len(data_list) == 0:
                    skip_time += 1
                    print("dataset is 0")
                    continue
                score_list, feature_list = orbit_grasper.predict(
                    data_list, sphere_indices_list, harmonics_list
                )  # [mask_num, n_points, m_intervals]

                score_list = (
                    torch.cat(score_list, dim=0).cpu().numpy()
                )  # [mask_num * n_points, m_intervals]
                # feature_list = torch.cat(feature_list, dim=0).cpu().numpy()  # [mask_num * n_points, 16]
                # a, b = divmod(score_list.argmax(), 36)
                # print(score_list[a])
                # print(feature_list[a])

                grasp_poses_list = torch.cat(grasp_poses_list, dim=0)
                # grasp_poses_list shape: [mask_num * n_points, m_intervals, 8]
                grasp_poses_list = (
                    torch.cat(
                        [grasp_poses_list[:, :, :3], grasp_poses_list[:, :, -4:]],
                        dim=-1,
                    )
                    .cpu()
                    .numpy()
                )

                grasp_normal_list = list()
                for k, feature_point in enumerate(data_list):
                    normal = feature_point.n
                    sphere_indices = sphere_indices_list[k]
                    normal = normal[sphere_indices]
                    grasp_normal_list.append(normal)
                grasp_normal_list = torch.cat(grasp_normal_list, dim=0).cpu().numpy()
                assert grasp_normal_list.shape[0] == score_list.shape[0]

                num_grasps = score_list.shape[0]
                all_positions = []
                all_rotations = []
                all_normals = []
                all_indices = []

                for l in range(num_grasps):
                    grasp_normals = np.array(
                        [grasp_normal_list[l]] * len(grasp_poses_list[l])
                    )
                    poses = np.array(grasp_poses_list[l])
                    positions = poses[:, :3]
                    rotations = poses[:, 3:]

                    all_positions.append(positions)
                    all_rotations.append(rotations)
                    all_normals.append(grasp_normals)
                    all_indices.append(np.array([[l, q] for q in range(len(poses))]))

                all_positions = np.vstack(all_positions)
                all_rotations = np.vstack(all_rotations)
                all_normals = np.vstack(all_normals)
                all_indices = np.vstack(all_indices)

                gripper_angle_masks = get_gripper_angle_mask_batch(
                    Rotation.from_quat(all_rotations)
                )

                invalid_indices = np.where(gripper_angle_masks == False)[0]
                score_list_flat = score_list.flatten()
                score_list_flat[invalid_indices] = -np.inf

                valid_indices = np.where(gripper_angle_masks)[0]

                valid_positions = all_positions[valid_indices]
                valid_rotations = all_rotations[valid_indices]
                valid_normals = all_normals[valid_indices]
                valid_score_list_flat = score_list_flat[valid_indices]

                valid_indices_2 = all_indices[valid_indices]
                actions = np.hstack((valid_positions, valid_rotations, valid_normals))
                graspable_results, position_after_translations, new_rotations = (
                    sim.decode_action_batch(actions, z_thresh=0.045)
                )

                valid_score_list_flat[graspable_results == 0] = -np.inf

                score_list_flat[valid_indices] = valid_score_list_flat
                score_list = score_list_flat.reshape(score_list.shape)

                valid_indices_2 = valid_indices_2[graspable_results != 0]
                position_after_translations = position_after_translations[
                    graspable_results != 0
                ]
                new_rotations = new_rotations[graspable_results != 0]

                filtered_grasp_poses = np.copy(grasp_poses_list)

                for idx, pos_after_trans, new_rot in zip(
                    valid_indices_2, position_after_translations, new_rotations
                ):
                    l, q = idx
                    trans_matrix = Transform(
                        Rotation.from_quat(new_rot), pos_after_trans
                    ).as_matrix()
                    z_mask = get_gripper_points_mask(trans_matrix, threshold=0.045)

                    if not z_mask:
                        score_list[l][q] = -np.inf
                    else:
                        filtered_grasp_poses[l][q] = np.concatenate(
                            [pos_after_trans, new_rot]
                        )

                if scene == "pile":
                    if score_list.max() < 3 and skip_time < 2:
                        print("no confident on this observation, skip")
                        skip_time += 1
                        continue
                    # best_pose_index_flat = np.argmax(score_list)
                    # best_pose_index = np.unravel_index(best_pose_index_flat, score_list.shape)
                    # best_grasp_pose = filtered_grasp_poses[best_pose_index[0]][best_pose_index[1]]
                    # # best_grasp_score = score_list[best_pose_index[0]][best_pose_index[1]]

                    while True:
                        best_pose_index_flat = np.argmax(score_list)
                        best_pose_index = np.unravel_index(
                            best_pose_index_flat, score_list.shape
                        )
                        best_grasp_pose = filtered_grasp_poses[best_pose_index[0]][
                            best_pose_index[1]
                        ]
                        finger_positions = sim.calculate_rectangle_vertices(
                            length=0.09,
                            width=0.018,
                            height=0.009,
                            tcp=best_grasp_pose[:3],
                            quat=best_grasp_pose[3:],
                        )
                        if np.all(finger_positions[:, 2] > 0.052):
                            break
                        else:
                            graspable, poses, rotations = modify_vertical_offset(
                                best_grasp_pose[:3],
                                best_grasp_pose[3:],
                                finger_positions,
                                0.007,
                                0.052,
                            )
                            if graspable:
                                best_grasp_pose = np.concatenate([poses, rotations])
                                break
                            score_list[best_pose_index] = -np.inf

                else:

                    candidate_score = np.where(score_list > 4)
                    candidate_grasp_poses = filtered_grasp_poses[candidate_score]
                    if len(candidate_grasp_poses) == 0:
                        cur_thresh = 4
                        while cur_thresh >= 3:
                            candidate_score = np.where(score_list > cur_thresh - 0.5)
                            if len(candidate_score[0]) > 0:
                                break
                            cur_thresh -= 0.5
                        candidate_grasp_poses = filtered_grasp_poses[candidate_score]

                    if len(candidate_grasp_poses) == 0:
                        skip_time += 1
                        print("no candidate grasp poses")
                        continue
                    # select the highest z
                    highest_z_grasp_pose = candidate_grasp_poses[
                        np.argmax(candidate_grasp_poses[:, 2])
                    ]
                    highest_z = highest_z_grasp_pose[2]
                    highest_z_score = score_list[candidate_score][
                        np.argmax(candidate_grasp_poses[:, 2])
                    ]
                    within_1cm_indices = np.where(
                        (highest_z - 0.01 <= candidate_grasp_poses[:, 2])
                        & (candidate_grasp_poses[:, 2] < highest_z)
                    )

                    if len(within_1cm_indices[0]) > 0:
                        within_1cm_poses = candidate_grasp_poses[within_1cm_indices]
                        within_1cm_scores = score_list[candidate_score][
                            within_1cm_indices
                        ]

                        max_within_1cm_score_idx = np.argmax(within_1cm_scores)
                        if (
                            within_1cm_scores[max_within_1cm_score_idx]
                            > highest_z_score
                        ):
                            best_grasp_pose = within_1cm_poses[max_within_1cm_score_idx]
                        else:
                            best_grasp_pose = highest_z_grasp_pose
                    else:
                        best_grasp_pose = highest_z_grasp_pose

                pos = best_grasp_pose[:3]
                rot = best_grasp_pose[3:]
                action = [pos, rot]
                grasp_success, width, description = sim.execute_grasp_fast(
                    action, remove=True, waitTime=2
                )

                cnt += 1
                if grasp_success:
                    success += 1
                    grasped_objs += 1
                    # print('Grasp success', 'best score:', score_list[best_pose_index[0]][best_pose_index[1]])
                else:
                    remain_objs += 1
                    # print('Grasp failed', 'best score:', score_list[best_pose_index[0]][best_pose_index[1]])

                if last_label == False and grasp_success == False:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 1

                last_label = grasp_success

            success_rate = 100.0 * success / cnt
            declutter_rate = 100.0 * success / total_objs

            print(
                "success grasp:",
                success,
                "total grasp:",
                cnt,
                "total objects:",
                total_objs,
                "grasped objs:",
                grasped_objs,
                "remain objs:",
                remain_objs,
            )
            print(
                "Grasp success rate: %.2f %%, Declutter rate: %.2f %%"
                % (success_rate, declutter_rate)
            )
            write_test(
                base_path / config_file["test"]["root_dir"],
                success,
                cnt,
                total_objs,
                grasped_objs,
                remain_objs,
                success_rate,
                declutter_rate,
                scene,
            )

        log = [success_rate, declutter_rate]
        success_list.append(success_rate)
        declutter_list.append(declutter_rate)

        record.append(log)
        write_log(
            base_path / config_file["test"]["root_dir"],
            success_rate,
            declutter_rate,
            scene,
        )

        sim.close()

    avg_success_rate = sum(success_list) / len(success_list)
    avg_declutter_rate = sum(declutter_list) / len(declutter_list)

    print("Average success", avg_success_rate, "Average declutter", avg_declutter_rate)


def modify_vertical_offset(
    position_after_translation, rot, finger_positions, vertical_offset, z_thresh
):
    rotation = Rotation.from_quat(rot)
    global_z_direction = rotation.apply([0, 0, 1])

    cur_vertical_offset = -5e-4
    while cur_vertical_offset >= -vertical_offset:
        z_translation_vector = global_z_direction * cur_vertical_offset
        final_position = position_after_translation + z_translation_vector

        final_finger_positions = finger_positions + z_translation_vector

        final_z = np.min(final_finger_positions[:, 2])

        if final_z >= z_thresh:
            return (1, final_position, rot)
        cur_vertical_offset -= 5e-4

    return (0, position_after_translation, rot)


def voxel_down_sample(original_pcd, voxel_size, gt_seg_arr=None):
    down_pcd, _, traced_indices = original_pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=original_pcd.get_min_bound(),
        max_bound=original_pcd.get_max_bound(),
    )

    new_normals = np.asarray(down_pcd.normals)
    norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    new_normals = new_normals / norms
    # down_pcd.normals = o3d.utility.Vector3dVector(new_normals)
    for i in range(len(new_normals)):
        point_z = down_pcd.points[i][2]
        if point_z > 0.065 or point_z < 0.051:
            continue

        original_normals = np.asarray(original_pcd.normals)[traced_indices[i]]

        if original_normals.size > 0:
            closest_z_normal = original_normals[
                np.abs(original_normals[:, 2]).argmin(), 2
            ]

            # if -0.2 < closest_z_normal < -0.1:
            #     closest_z_normal += 0.1
            # elif 0.1 < closest_z_normal < 0.2:
            #     closest_z_normal -= 0.1

            x, y, _ = new_normals[i]
            xy_squared = x**2 + y**2
            if xy_squared > 1e-8:
                alpha = np.sqrt((1 - closest_z_normal**2) / xy_squared)
                down_pcd_normal_xy = np.array([x * alpha, y * alpha])
            else:
                down_pcd_normal_xy = np.array([0, 0])
            new_normal = np.append(down_pcd_normal_xy, closest_z_normal)
            new_normals[i] = new_normal

    # Update normals in the downsampled point cloud
    down_pcd.normals = o3d.utility.Vector3dVector(new_normals)

    if gt_seg_arr is not None:
        down_pcd_gts = np.zeros(len(down_pcd.points), dtype=np.uint8)
        for i, indices in enumerate(traced_indices):
            temp = gt_seg_arr[indices]
            counter = Counter(temp)
            main_seg_idx, _ = counter.most_common(1)[0]

            down_pcd_gts[i] = main_seg_idx

    if gt_seg_arr is not None:
        return down_pcd, down_pcd_gts
    return down_pcd


def quaternion_to_z_direction(quaternions):
    quaternions_np = quaternions.cpu().numpy()
    rotations = Rotation.from_quat(quaternions_np.reshape(-1, 4))
    z_axis = np.array([0, 0, -1])
    z_rotated = rotations.apply(z_axis)
    z_rotated_tensor = torch.from_numpy(z_rotated).view(
        quaternions_np.shape[0], quaternions_np.shape[1], 3
    )

    return z_rotated_tensor.detach()


def vector_to_spherical(vectors):
    r = torch.sqrt(torch.sum(vectors**2, dim=-1))

    theta = torch.acos(vectors[..., 1] / r)
    phi = torch.atan2(vectors[..., 0], vectors[..., 2])
    return r, theta, phi


def compute_spherical_harmonics(vectors, lmax=3):
    r, theta, phi = vector_to_spherical(vectors)
    harmonics_list = []
    for l in range(lmax + 1):
        harmonics = spherical_harmonics_alpha_beta(
            l, phi, theta, normalization="component"
        )
        harmonics_list.append(harmonics)
    harmonics = torch.cat(harmonics_list, dim=-1)
    return harmonics.detach()


def normalize(data: FeaturedPoints):
    pos = data.x
    center = pos.mean(dim=0, keepdim=True)
    pos = pos - center
    normalized_data = FeaturedPoints(x=pos, n=data.n, b=data.b)
    return normalized_data


def get_gripper_angle_mask_batch(rotations, threshold=75):
    global_z_directions = rotations.apply(np.array([[0, 0, 1]] * len(rotations)))
    z_axis = np.array([0, 0, 1])
    cos_angles = np.dot(global_z_directions, z_axis)
    cos_angles = np.clip(cos_angles, -1, 1)
    angles_rad = np.arccos(cos_angles)
    angles_deg = np.degrees(angles_rad)
    assert np.any(angles_deg > threshold) and np.any(angles_deg <= threshold)
    return angles_deg > threshold


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def find_checkpoint(root_dir, prefix):
    files = os.listdir(root_dir)
    matching_files = [f for f in files if f"-ckpt-{prefix}-" in f and f.endswith(".pt")]
    if not matching_files:
        raise ValueError(f"No checkpoints found with prefix '{prefix}' in '{root_dir}'")
    matching_files.sort(reverse=True)
    return matching_files[0]


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "single_config.yaml"
    # config_path = "./training_config.yaml"
    config = load_config(config_path)

    SHOW_PCD = False
    sys.setrecursionlimit(500000)
    np.set_printoptions(threshold=np.inf)
    grasp_test_random(config_file=config)
