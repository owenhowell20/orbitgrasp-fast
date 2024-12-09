import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.transform import Rotation
from e3nn.o3 import spherical_harmonics_alpha_beta
import argparse
from torch_geometric.nn import radius


class save_harmonics():
    def __init__(self, path, lmax=1, camera_setting='single'):
        # self.data: training data -> FeaturedPoints
        # self.scenes_grasping_list: labels -> for each scene
        super().__init__()
        self.lmax = lmax
        self.harmonics_list = list()
        files = sorted([f for f in os.listdir(path) if f.endswith('.pkl')])
        for itr, file in enumerate(files):
            # load the file (each scene)
            with open(os.path.join(path, file), 'rb') as f:
                scene = pickle.load(f)
            pcd_points = torch.from_numpy(scene.get('pcd_points')).float()
            down_mask_list = scene.get('all_masks')
            all_poses_info_list = scene.get('grasping_pose')

            flag = 0
            harmonics_list = list()

            for m, mask in enumerate(down_mask_list):
                local_points = pcd_points[mask]
                if local_points.shape[0] == 0:
                    continue
                grasp_poses = torch.from_numpy(all_poses_info_list[flag]['actions'])
                order = all_poses_info_list[flag]['order']

                grasp_quats = grasp_poses[:, :, -4:]

                grasp_z = self.quaternion_to_z_direction(grasp_quats)
                grasp_harmonics = self.compute_spherical_harmonics(grasp_z)
                harmonics_list.append(grasp_harmonics)
                flag += 1
                assert order == m

            self.harmonics_list.append(harmonics_list)
        torch.save(self.harmonics_list, f'scripts/grasp_harmonics_l{lmax}_test_{camera_setting}.pt')
        print(f'grasp_harmonics_l{lmax}_test_{camera_setting}.pt saved')

    def quaternion_to_z_direction(self, quaternions):
        quaternion_np = quaternions.numpy()
        rotations = Rotation.from_quat(quaternion_np.reshape(-1, 4))
        z_axis = np.array([0, 0, -1])
        z_rotated = rotations.apply(z_axis)
        z_rotated_tensor = torch.from_numpy(z_rotated).view(quaternions.shape[0], quaternions.shape[1], 3).to(
            quaternions.device)

        return z_rotated_tensor

    def vector_to_spherical(self, vectors):

        r = torch.sqrt(torch.sum(vectors ** 2, dim=-1))

        theta = torch.acos(vectors[..., 1] / r)
        phi = torch.atan2(vectors[..., 0], vectors[..., 2])
        return r, theta, phi

    def compute_spherical_harmonics(self, vectors):
        r, theta, phi = self.vector_to_spherical(vectors)
        harmonics_list = []
        alpha = phi
        beta = theta
        for l in range(self.lmax + 1):
            harmonics = spherical_harmonics_alpha_beta(l, alpha, beta, normalization='component')
            harmonics_list.append(harmonics)
        harmonics = torch.cat(harmonics_list, dim=-1)
        return harmonics


class save_harmonics_wo_mask():
    def __init__(self, path, lmax=3, camera_setting='single'):
        # self.data: training data -> FeaturedPoints
        # self.scenes_grasping_list: labels -> for each scene
        super().__init__()
        self.lmax = lmax
        self.harmonics_list = list()
        files = sorted([f for f in os.listdir(path) if f.endswith('.pkl')])
        for itr, file in enumerate(files):
            # load the file (each scene)
            with open(os.path.join(path, file), 'rb') as f:
                scene = pickle.load(f)
            pcd_points = torch.from_numpy(scene.get('pcd_points')).float()
            sample_indices = scene.get('sample_index')
            all_poses_info_list = scene.get('grasping_pose')

            flag = 0
            harmonics_list = list()

            down_pcd_points_tensor = pcd_points.clone().detach()
            for m, center_index in enumerate(sample_indices):
                center_point = pcd_points[center_index]
                center_point = center_point.clone().detach().reshape(1, 3)
                edge_index = radius(down_pcd_points_tensor, center_point, r=0.04, max_num_neighbors=900)
                local_points = pcd_points[edge_index[1]]
                if local_points.shape[0] == 0:
                    continue
                grasp_poses = torch.from_numpy(all_poses_info_list[flag]['actions'])
                order = all_poses_info_list[flag]['order']

                grasp_quats = grasp_poses[:, :, -4:]

                grasp_z = self.quaternion_to_z_direction(grasp_quats)
                grasp_harmonics = self.compute_spherical_harmonics(grasp_z)
                harmonics_list.append(grasp_harmonics)
                flag += 1
                assert order == m

            self.harmonics_list.append(harmonics_list)
        torch.save(self.harmonics_list, f'scripts/grasp_harmonics_l{lmax}_test_{camera_setting}_wo_mask.pt')
        print(f'grasp_harmonics_l{lmax}_test_{camera_setting}_wo_mask.pt saved')

    def quaternion_to_z_direction(self, quaternions):
        quaternion_np = quaternions.numpy()
        rotations = Rotation.from_quat(quaternion_np.reshape(-1, 4))
        z_axis = np.array([0, 0, -1])
        z_rotated = rotations.apply(z_axis)
        z_rotated_tensor = torch.from_numpy(z_rotated).view(quaternions.shape[0], quaternions.shape[1], 3).to(
            quaternions.device)

        return z_rotated_tensor

    def vector_to_spherical(self, vectors):

        r = torch.sqrt(torch.sum(vectors ** 2, dim=-1))

        theta = torch.acos(vectors[..., 1] / r)
        phi = torch.atan2(vectors[..., 0], vectors[..., 2])
        return r, theta, phi

    def compute_spherical_harmonics(self, vectors):
        r, theta, phi = self.vector_to_spherical(vectors)
        harmonics_list = []
        alpha = phi
        beta = theta
        for l in range(self.lmax + 1):
            harmonics = spherical_harmonics_alpha_beta(l, alpha, beta, normalization='component')
            harmonics_list.append(harmonics)
        harmonics = torch.cat(harmonics_list, dim=-1)
        return harmonics


if __name__ == '__main__':
    def strToBool(value):
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')


    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_setting', type=str, default='single')
    parser.add_argument('--test_path', type=str,
                        default=Path(__file__).resolve().parent.parent / 'dataset' / 'collected_data/se3_filtered')
    parser.add_argument('--use_mask', type=strToBool, default=True)
    args = parser.parse_args()
    camera_setting = args.camera_setting
    path = args.test_path / f'test_{camera_setting}'

    if args.use_mask:
        save_harmonics(path=path, lmax=3, camera_setting=camera_setting)
    else:
        save_harmonics_wo_mask(path=path, lmax=3, camera_setting=camera_setting)
