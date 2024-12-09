import os
import pickle
import numpy as np

from pathlib import Path
from generate_pose_multi_wo_mask import get_gripper_angle_mask
from utils.transform import Rotation, Transform
from utils.utility import get_gripper_points_mask
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='pile')
    parser.add_argument('--camera', type=str, default='single')
    parser.add_argument('--save_path', type=str,
                        default=Path(__file__).resolve().parent / 'collected_data/se3_filtered')

    parser.add_argument('--read_path', type=str,
                        default=Path(__file__).resolve().parent / 'collected_data/se3_origin')

    args = parser.parse_args()

    scene_name = args.scene
    read_path = args.read_path / f'{scene_name}_{args.camera}'
    save_path = args.save_path / f'{scene_name}_{args.camera}'
    files = sorted([f for f in os.listdir(read_path) if f.endswith('.pkl')])


    def calculate_rectangle_vertices(length, width, height, tcp, quat):
        rotation_matrix = Rotation.from_quat(quat).as_matrix()

        vectors_local = [
            np.array([+width / 2, +length / 2, height]),
            np.array([-width / 2, +length / 2, height]),
            np.array([-width / 2, -length / 2, height]),
            np.array([+width / 2, -length / 2, height]),
        ]

        vertices = []
        for v in vectors_local:
            v_rotated = rotation_matrix.dot(v) + tcp
            vertices.append(v_rotated)

        return np.asarray(vertices)


    vertical_offset = 0.007
    translation_offset = 0.008
    z_thresh = 0.052

    total_grasp = 0
    success_grasp = 0
    failure_grasp = 0
    unreachable_grasp = 0

    success_grasp_ori = 0
    failure_grasp_ori = 0
    unreachable_grasp_ori = 0

    c = 0
    for itr, file in enumerate(files):

        # load the file (each scene)
        with open(os.path.join(read_path, file), 'rb') as f:
            scene = pickle.load(f)

        success_flag_list = scene.get('success')
        action_all_masks = scene.get('grasping_pose')

        success_array = list()
        grasp_indices_array = list()
        new_success_flag_list = list()

        c += len(action_all_masks)
        for m in range(len(action_all_masks)):
            # for each mask, we need the success flag and its corresponding grasping pose
            success_flag = success_flag_list[m]
            new_success_flag = np.zeros((success_flag.shape[0], success_flag.shape[1], success_flag.shape[2])).astype(
                np.float32)

            actions = action_all_masks[m]['actions']
            normals = action_all_masks[m]['normals']

            success_list = list()

            for i, normal in enumerate(normals):
                success_arr = success_flag[i]
                success_arr = success_arr.astype(int)
                assert len(actions[i]) == len(success_arr)
                total_grasp += len(actions[i])
                for k, action in enumerate(actions[i]):

                    pos = action[:3]
                    rot = action[-4:]

                    gripper_angle_mask = get_gripper_angle_mask(Rotation.from_quat(rot), threshold=75)

                    if gripper_angle_mask:

                        orig_position, rot, normal_vector = pos, rot, normal

                        unit_normal_vector = normal_vector / np.linalg.norm(normal_vector)

                        translation_distance = 0.04 - translation_offset
                        translation_vector = -unit_normal_vector * translation_distance
                        position_after_translation = orig_position + translation_vector

                        finger_positions = calculate_rectangle_vertices(length=0.09, width=0.018, height=0.009,
                                                                        tcp=position_after_translation,
                                                                        quat=rot)

                        if np.all(finger_positions[:, 2] > z_thresh):
                            continue

                        rotation = Rotation.from_quat(rot)
                        global_z_direction = rotation.apply([0, 0, 1])

                        cur_vertical_offset = -5e-4
                        while cur_vertical_offset >= -vertical_offset:
                            z_translation_vector = global_z_direction * cur_vertical_offset
                            final_position = position_after_translation + z_translation_vector

                            final_finger_positions = finger_positions + z_translation_vector

                            final_z = np.min(final_finger_positions[:, 2])

                            if final_z >= z_thresh:
                                break
                            cur_vertical_offset -= 5e-4

                        if cur_vertical_offset < -vertical_offset:
                            success_arr[k] = -1
                        trans_matrix = Transform(Rotation.from_quat(rot),
                                                 position_after_translation).as_matrix()
                        z_mask = get_gripper_points_mask(trans_matrix, threshold=z_thresh)
                        if not z_mask:
                            success_arr[k] = -1

                    else:
                        success_arr[k] = -1
                success_grasp += np.sum(success_arr == 1)
                failure_grasp += np.sum(success_arr == 0)
                unreachable_grasp += np.sum(success_arr == -1)
                new_success_flag[i] = success_arr

                success_grasp_ori += np.sum(success_flag[i] == 1)
                failure_grasp_ori += np.sum(success_flag[i] == 0)
                unreachable_grasp_ori += np.sum(success_flag[i] == -1)
            new_success_flag_list.append(new_success_flag)

        scene['success'] = new_success_flag_list
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, f'{scene_name}_{(itr):05}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(scene, f)
        print('Total grasp:', total_grasp, 'Success grasp:', success_grasp, 'Failure grasp:', failure_grasp,
              'Unreachable grasp:', unreachable_grasp)
        print('Total grasp ori:', total_grasp, 'Success grasp ori:', success_grasp_ori, 'Failure grasp ori:',
              failure_grasp_ori,
              'Unreachable grasp ori:', unreachable_grasp_ori)

    print(c)
