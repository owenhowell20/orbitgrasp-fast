import numpy as np
import sys
import os
from math import sin, cos

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from simulator.simulation_clutter_bandit_single_camera import ClutterRemovalSim
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from simulator.sam3d import filter_masks
from utils.transform import Transform, Rotation


def show_anns(anns, ax, draw_bbox=False):
    if len(anns) == 0:
        return

    ax.set_autoscale_on(False)
    rng = np.random.RandomState()

    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0

    for ann in anns:
        m = ann['segmentation']
        rgb_color = rng.random(3).astype(np.float32)
        alpha = np.asarray([0.35]).astype(np.float32)
        color_mask = np.concatenate([rgb_color, alpha])
        img[m] = color_mask
        if draw_bbox:
            x, y, w, h = ann['bbox']

            bbox = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=rgb_color, facecolor='None')
            ax.add_patch(bbox)
    ax.imshow(img)


def run(scene, start=0, iteration_num=200, lambda_p=4, gui=True, device='cuda:0'):
    sim = ClutterRemovalSim(scene=scene, object_set='train', rand=True, gui=gui, seed=12345)
    total_num = 0
    total_mask = 0

    sam_checkpoint = Path(__file__).resolve().parent.parent / 'pretrained/sam_vit_h_4b8939.pth'
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    for i in range(iteration_num):

        objs_num = sim.rng.poisson(lambda_p) + 1
        r = np.random.uniform(1.5, 2) * sim.size
        theta = np.random.uniform(np.pi / 4, np.pi / 2.4)
        phi = np.random.uniform(0.0, np.pi)
        origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0 + 0.15])
        eye = np.r_[
            r * sin(theta) * cos(phi),
            r * sin(theta) * sin(phi),
            r * cos(theta),
        ]
        eye = eye + origin.translation

        a = Transform.look_at(eye - origin.translation, np.array([0, 0, 0]), [0, 0, 1]) * origin.inverse()
        a = a.inverse().to_list()
        camera_config = {
            'image_size': (480, 640),
            'intrinsics': (450., 0, 320., 0, 450., 240., 0, 0, 1),
            'position': eye,
            'rotation': a[:4],
            'zrange': (0.01, 2.),
            'noise': False,
            'name': 'random'
        }

        #######################################################################################
        # original camera setting of edge grasp.
        # ours is slightly different with their setting.
        # Meanwhile, the camera intrinsic is different, so if you need to generate grasp pose
        # on the original setting, please use the following setting and camera intrinsic.
        # NOTE: remember to change the camera intrinsic in the generate_pose_single.py

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
        indexes, pose_list, color_list = sim.reset(objs_num)
        sim.getRandomViewObservation(camera_config)

        colormaps = sim.colormaps.astype(np.uint8)
        depthmaps = sim.heightmaps

        masks_origin = mask_generator.generate(colormaps[0])

        if show_mask:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(colormaps[0])
            masks = filter_masks(masks_origin, sim.pcds_xyz, 0.3,
                                 bound_ratio=sim.bound_radio)
            print(len(masks))
            show_anns(masks, ax, draw_bbox=True)
            ax.set_title("Bounding Boxes of Filtered Masks", fontsize=30)
            ax.axis('off')
            plt.show()

        # save
        if save_data:
            if not os.path.exists(Path(__file__).resolve().parent / f'store/RawData_{scene}_single'):
                os.makedirs(Path(__file__).resolve().parent / f'store/RawData_{scene}_single')
                os.makedirs(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Images')
                os.makedirs(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Masks')
                os.makedirs(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Indexes')
                os.makedirs(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Poses')
                os.makedirs(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Colors')
                os.makedirs(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Camera')

            np.save(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Images/rgb_{i + start:05}.npy',
                    colormaps)
            np.save(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Images/depth_{i + start:05}.npy',
                    depthmaps)
            with open(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Masks/mask_{i + start:05}.pkl',
                      'wb') as f:
                pickle.dump(masks_origin, f)  # type: ignore

            with open(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Indexes/index_{i + start:05}.pkl',
                      'wb') as f:
                pickle.dump(indexes, f)  # type: ignore

            with open(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Poses/pose_{i + start:05}.pkl',
                      'wb') as f:
                pickle.dump(pose_list, f)  # type: ignore
            with open(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Colors/color_{i + start:05}.pkl',
                      'wb') as f:
                pickle.dump(color_list, f)  # type: ignore
            with open(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Camera/camera_{i + start:05}.pkl',
                      'wb') as f:
                pickle.dump(camera_config, f)  # type: ignore

        print(
            f"Iteration {i + start} is saved, please check. Total Object num is: {objs_num}. Origin Masks num is {len(masks_origin)}")
        total_num += objs_num
        total_mask += len(masks_origin)

        print(f"total num is: {total_num}, total mask is: {total_mask}")
    sim.close()


def test(scene, i, from_save=True, gui=True):
    with open(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Indexes/index_{i:05}.pkl', 'rb') as f:
        while True:
            try:
                index = pickle.load(f)
            except EOFError:  # End of File
                break

    # print(index)

    with open(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Poses/pose_{i:05}.pkl', 'rb') as f:
        while True:
            try:
                pose = pickle.load(f)
            except EOFError:  # End of File
                break
    # print(pose)

    with open(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Masks/mask_{i:05}.pkl', 'rb') as f:
        while True:
            try:
                mask_list = pickle.load(f)
            except EOFError:  # End of File
                break

    with open(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Colors/color_{i:05}.pkl', 'rb') as f:
        while True:
            try:
                color = pickle.load(f)
            except EOFError:  # End of File
                break

    with open(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Camera/camera_{i:05}.pkl', 'rb') as f:
        while True:
            try:
                camera_config = pickle.load(f)
            except EOFError:  # End of File
                break

    env = ClutterRemovalSim(scene=scene, object_set='train', rand=True, gui=gui)

    objs_num = len(index)
    env.reset(objs_num, color_=color, pose_=pose, index=index, from_save=from_save)
    env.getRandomViewObservation(camera_config)

    colormaps = env.colormaps.astype(np.uint8)
    depthmaps = env.heightmaps

    mask_num_ori = 0
    mask_num_filtered = 0

    mask_num_ori += len(mask_list)
    fig, axs = plt.subplots(1, 4, figsize=(40, 20))
    axs[0].imshow(colormaps[0])
    axs[0].set_title("new Image", fontsize=30)
    axs[0].axis('off')

    img = np.load(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Images/rgb_{i:05}.npy')
    img_depth = np.load(Path(__file__).resolve().parent / f'store/RawData_{scene}_single/Images/depth_{i:05}.npy')

    print("Equal Depth Value:", np.array_equal(depthmaps, img_depth))
    print("Equal RGB Value:", np.array_equal(colormaps, img))

    if show_mask:

        axs[1].imshow(img[0])
        axs[1].set_title("original Image", fontsize=30)
        axs[1].axis('off')

        axs[2].imshow(colormaps[0])
        masks_filtered = filter_masks(mask_list, env.pcds_xyz, 0.3,
                                      bound_ratio=env.bound_radio)
        mask_num_filtered += len(masks_filtered)
        show_anns(masks_filtered, axs[2], draw_bbox=True)
        axs[2].set_title("Bounding Boxes of Filtered Masks", fontsize=30)
        axs[2].axis('off')

        axs[3].imshow(colormaps[0])
        show_anns(mask_list, axs[3], draw_bbox=True)
        axs[3].set_title("Bounding Boxes of Origin Masks", fontsize=30)
        axs[3].axis('off')
        plt.savefig("test.png")
        plt.show()

    num_bodies = env.world.p.getNumBodies()
    print("mask_num_ori:", mask_num_ori)
    print("mask_num_filtered:", mask_num_filtered)
    env.close()


if __name__ == '__main__':
    sys.setrecursionlimit(5000000)
    np.set_printoptions(threshold=np.inf)


    def strToBool(value):
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')


    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default="pile")
    parser.add_argument('--save_data', type=strToBool, default=True)
    parser.add_argument('--show_mask', type=strToBool, default=False)
    parser.add_argument('--GUI', type=strToBool, default=False)
    parser.add_argument('--lambda_p', type=int, default=4)
    parser.add_argument('--sample_num', type=int, default=2000)

    args = parser.parse_args()

    scene = args.scene
    save_data = args.save_data
    show_mask = args.show_mask
    GUI = args.GUI
    LAMBDA_P = args.lambda_p
    sample_num = args.sample_num
    run(scene, 0, sample_num, lambda_p=LAMBDA_P, gui=GUI)

    # TEST generation
    # for i in range(0, 5):
    #     test(scene, i, gui=GUI)
