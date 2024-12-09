import numpy as np
import torch


def downsample_points(pts, K):
    # if num_pts > 2K use farthest sampling
    # else use random sampling
    if pts.shape[0] >= 2 * K:
        sampler = FarthestSampler()
        return sampler(pts, K)
    else:
        return pts[np.random.choice(pts.shape[0], K, replace=(K < pts.shape[0])), :]


class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        index_list = []
        farthest_pts = np.zeros((k, 3), dtype=np.float32)
        index = np.random.randint(len(pts))
        farthest_pts[0] = pts[index]
        index_list.append(index)
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            index = np.argmax(distances)
            farthest_pts[i] = pts[index]
            index_list.append(index)
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts, index_list


class FarthestSamplerTorch:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        index_list = []
        farthest_pts = torch.zeros(k, 3).to(pts.device)
        index = np.random.randint(len(pts))
        farthest_pts[0] = pts[index]
        index_list.append(index)
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            index = torch.argmax(distances)
            farthest_pts[i] = pts[index]
            index_list.append(index)
            distances = torch.minimum(distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts, index_list


def get_gripper_points(trans):
    gripper_points_sim = np.array([
        [0.012, 0.09, -0.04],
        [0.012, -0.09, -0.04],
        [-0.012, 0.09, -0.04],
        [-0.012, -0.09, -0.04],

        [0.018, 0.09, -0.11],
        [0.018, -0.09, -0.11],
        [-0.018, 0.09, -0.11],
        [-0.018, -0.09, -0.11]])

    gripper_points_sim = trans[:3, :3] @ gripper_points_sim.transpose()
    gripper_points_sim = gripper_points_sim.transpose()
    # print(gripper_points_sim.size())
    gripper_points_sim = gripper_points_sim + trans[:3, -1]
    return gripper_points_sim


def get_gripper_points_mask(trans, threshold=0.052):
    gripper_points_sim = get_gripper_points(trans)
    z_value = gripper_points_sim[:, -1]
    z_mask = z_value > threshold
    z_mask = np.all(z_mask)
    return z_mask
