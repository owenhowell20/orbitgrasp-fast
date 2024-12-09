import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.FeaturedPoints import FeaturedPoints
from utils.transform import Rotation
from e3nn.o3 import spherical_harmonics_alpha_beta

"""
data_save(filename=filename, mode=mode, pcd_points=down_pcd_points,
                      pcd_normals=down_pcd_normals, all_masks=down_mask_list,
                      grasping_pose=action_all_masks, success=success_list_all, object_num=len(env.world.bodies_list))

action_info = {'actions': actions_list, 'points': points_list, 'normals': normals_list,
                'indices': indices_list, 'grasp_indices': select_idx[indices_list]}
action_all_masks.append(action_info)
"""


class orbitgrasp_dataset(Dataset):
    def __init__(self, path, extra_radius=0.035, lmax=3, max_allowed_points=800,
                 min_allowed_points=700, min_radius=-1e-6, augment=False, load_harmonics=False, load_harmony_path='grasp_harmonics_l3_test.pt'):
        # self.data: training data -> FeaturedPoints
        # self.scenes_grasping_list: labels -> for each scene
        super().__init__()
        self.lmax = lmax
        self.data = list()
        self.scenes_grasping_indices = list()
        self.success_list = list()
        self.grasp_poses_list = list()
        self.harmonics_list = list()
        self.initial_radius = extra_radius
        self.aug_data = []

        if augment:
            self.augmentor = GraspAugmentation(lmax=lmax, augment_ratio=0)
        else:
            if load_harmonics:
                grasp_harmonics = torch.load(load_harmony_path)
                self.augmentor = None
            else:
                self.augmentor = GraspAugmentation(lmax=lmax, augment_ratio=0.)
        files = sorted([f for f in os.listdir(path) if f.endswith('.pkl')])

        for itr, file in enumerate(files):

            # load the file (each scene)
            with open(os.path.join(path, file), 'rb') as f:
                scene = pickle.load(f)
            pcd_points = torch.from_numpy(scene.get('pcd_points')).float()
            pcd_normals = torch.from_numpy(scene.get('pcd_normals')).float()
            down_mask_list = scene.get('all_masks')
            success_flag_list = scene.get('success')
            all_poses_info_list = scene.get('grasping_pose')

            if load_harmonics and not augment:
                scene_harmonics = grasp_harmonics[itr]
            else:
                scene_harmonics = None

            flag = 0
            for mask in down_mask_list:
                local_points = pcd_points[mask]
                if local_points.shape[0] == 0:
                    continue
                mask_center = local_points.mean(dim=0)
                distances = torch.norm(local_points - mask_center, dim=1)

                # Find all points within this expanded radius from the centroid
                temp_radius = self.initial_radius
                itr = 0
                while True:
                    mask_radius = distances.max() + temp_radius
                    all_distances_to_center = torch.norm(pcd_points - mask_center, dim=1)
                    sphere_mask = all_distances_to_center <= mask_radius
                    sphere_points = pcd_points[sphere_mask]

                    if sphere_points.shape[0] <= max_allowed_points and sphere_points.shape[0] >= min_allowed_points:
                        break
                    elif sphere_points.shape[0] > max_allowed_points:
                        temp_radius -= 0.005
                        if temp_radius <= min_radius:
                            break
                    elif sphere_points.shape[0] < min_allowed_points:
                        temp_radius += 0.005
                        if temp_radius > 0.15:
                            break
                    itr += 1
                    if itr >= 50:
                        break

                grasp_indices = torch.from_numpy(all_poses_info_list[flag]['grasp_indices'])
                grasp_poses = torch.from_numpy(all_poses_info_list[flag]['actions']).float()

                success_flag = np.squeeze(success_flag_list[flag])  # [P, I]
                # Map pcd_points indices to local indices within the sphere
                local_indices = torch.arange(pcd_points.shape[0])[sphere_mask]
                # get the local indices in the sphere points
                sphere_indices, reorderded_indices = torch.where(
                    local_indices.unsqueeze(1) == grasp_indices.unsqueeze(0))
                reorder_success_flag = success_flag[reorderded_indices]
                reordered_grasp_poses = grasp_poses[reorderded_indices]

                if scene_harmonics:
                    mask_harmonics = scene_harmonics[flag]
                    reorder_mask_harmonics = mask_harmonics[reorderded_indices]

                if sphere_points.shape[0] > (max_allowed_points + 100) or sphere_points.shape[0] < (
                        min_allowed_points - 100):
                    flag += 1
                    continue
                # visualize the sphere points
                # open3d_points = o3d.geometry.PointCloud()
                # open3d_points.points = o3d.utility.Vector3dVector(sphere_points.numpy())
                # o3d.visualization.draw_geometries([open3d_points])

                feature_points = FeaturedPoints(
                    x=sphere_points,
                    n=pcd_normals[sphere_mask],
                    b=torch.ones(sphere_points.shape[0], dtype=torch.long))

                feature_points = self.normalize(feature_points)

                self.data.append(feature_points)
                self.scenes_grasping_indices.append(sphere_indices)
                self.success_list.append(torch.from_numpy(reorder_success_flag).float())
                self.grasp_poses_list.append(reordered_grasp_poses)
                if scene_harmonics:
                    self.harmonics_list.append(reorder_mask_harmonics)
                flag += 1
        print('Data is loaded from: ', path,
              'Number of scenes: ', len(files),
              'Number of masks: ', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if not self.aug_data:
            data = self.data[idx]
            scene_grasping_indices = self.scenes_grasping_indices[idx]
            success_list = self.success_list[idx]
            grasp_poses = self.grasp_poses_list[idx]
            if self.augmentor:
                return self.augmentor(data, scene_grasping_indices, success_list, grasp_poses)
            return data, scene_grasping_indices, success_list, self.harmonics_list[idx]
        else:
            return self.aug_data[idx], self.aug_grasping_indices[idx], self.aug_success_list[idx], self.aug_harmonics[
                idx]

    def normalize(self, data: FeaturedPoints):
        pos = data.x
        center = pos.mean(dim=0, keepdim=True)
        pos = pos - center
        normalized_data = FeaturedPoints(x=pos, n=data.n, b=data.b)
        return normalized_data

    def aug_data_before_epoch(self):
        self.aug_data = list()
        self.aug_grasping_indices = list()
        self.aug_success_list = list()
        self.aug_harmonics = list()
        for idx in range(len(self.data)):
            data = self.data[idx]
            scene_grasping_indices = self.scenes_grasping_indices[idx]
            success_list = self.success_list[idx]
            grasp_poses = self.grasp_poses_list[idx]

            augmented_data, augmented_scene_grasping_indices, augmented_success_list, augmented_harmonics = (
                self.augmentor(data, scene_grasping_indices, success_list, grasp_poses))
            self.aug_data.append(augmented_data)
            self.aug_grasping_indices.append(augmented_scene_grasping_indices)
            self.aug_success_list.append(augmented_success_list)
            self.aug_harmonics.append(augmented_harmonics)
        assert len(self.aug_data) == len(self.data)


class GraspAugmentation:

    def __init__(self, lmax=3, angle_range=(-180, 180), augment_ratio=0.5):
        self.x_angle_range = (angle_range[0] / 2 + 2, angle_range[1] / 2 - 2)
        self.y_angle_range = (angle_range[0] / 2 + 2, angle_range[1] / 2 - 2)
        self.z_angle_range = angle_range
        self.augment_ratio = 0.
        self.lmax = lmax

    def __call__(self, data, scene_grasping_indices, success_list, grasp_poses):
        return self.process_without_rotation(data, scene_grasping_indices, success_list, grasp_poses)

    def process_without_rotation(self, data, scene_grasping_indices, success_list, grasp_poses):
        grasp_quats = grasp_poses[:, :, -4:]
        grasp_z = self.quaternion_to_z_direction(grasp_quats)
        mask_harmonics = self.compute_spherical_harmonics(grasp_z)
        return data, scene_grasping_indices, success_list, mask_harmonics

    def quaternion_to_z_direction(self, quaternions):
        rotations = Rotation.from_quat(quaternions.reshape(-1, 4))
        z_axis = np.array([0, 0, -1])
        z_rotated = rotations.apply(z_axis)
        z_rotated_tensor = torch.from_numpy(z_rotated).view(quaternions.shape[0], quaternions.shape[1], 3)

        return z_rotated_tensor.detach()

    def vector_to_spherical(self, vectors):
        r = torch.sqrt(torch.sum(vectors ** 2, dim=-1))

        theta = torch.acos(vectors[..., 1] / r)
        phi = torch.atan2(vectors[..., 0], vectors[..., 2])
        return r, theta, phi

    def compute_spherical_harmonics(self, vectors):
        r, theta, phi = self.vector_to_spherical(vectors)
        harmonics_list = []
        for l in range(self.lmax + 1):
            harmonics = spherical_harmonics_alpha_beta(l, phi, theta, normalization='component')
            harmonics_list.append(harmonics)
        harmonics = torch.cat(harmonics_list, dim=-1)
        return harmonics.detach()
