import matplotlib.patches as patches
import numpy as np


def show_anns(masks, ax, draw_bbox=True):
    if len(masks) == 0:
        return

    ax.set_autoscale_on(False)
    rng = np.random.RandomState()

    img = np.ones(
        (masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 4)
    )
    img[:, :, 3] = 0

    for mask in masks:
        m = mask["segmentation"]
        rgb_color = rng.random(3).astype(np.float32)
        alpha = np.asarray([0.35]).astype(np.float32)
        color_mask = np.concatenate([rgb_color, alpha])
        img[m] = color_mask
        if draw_bbox:
            x, y, w, h = mask["bbox"]

            bbox = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor=rgb_color, facecolor="None"
            )
            ax.add_patch(bbox)
    ax.imshow(img)


def compute_iou(mask_1, mask_2):
    mask1 = mask_1.get("segmentation")
    mask2 = mask_2.get("segmentation")
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    else:
        return intersection / union


def filter_masks(
    masks_list,
    pcds_xyz,
    iou_threshold=0.2,
    z_threshold=0.053,
    bound_ratio=1.5,
    offset=0,
    width_norm=48,
    height_norm=40,
):
    if len(masks_list) == 0:
        return []

    mask_shape = masks_list[0]["segmentation"].shape

    width_bound = [mask_shape[0] / (width_norm + offset), mask_shape[0] / bound_ratio]
    height_bound = [mask_shape[1] / (height_norm + offset), mask_shape[1] / bound_ratio]

    non_zero_indices = ~np.all(pcds_xyz == 0, axis=1)

    filtered_masks = []

    for mask in masks_list:

        mask_points = pcds_xyz[non_zero_indices] * (
            mask["segmentation"].reshape(-1, 1)[non_zero_indices]
        )
        mask_points = mask_points[~np.all(mask_points == 0, axis=1)]
        if mask_points.shape[0] == 0 or mask_points[:, -1].mean() <= z_threshold:
            continue

        delta_width = mask["bbox"][2]
        delta_height = mask["bbox"][3]

        if (
            delta_height < height_bound[1]
            and delta_width < width_bound[1]
            and (delta_height > height_bound[0] and delta_width > width_bound[0])
        ):
            keep = True

            for filtered_mask in filtered_masks:
                if compute_iou(mask, filtered_mask) > iou_threshold:
                    keep = False
                    break

            if keep:
                filtered_masks.append(mask)

    return filtered_masks


def sort_mask(masks):
    sorted_masks = sorted(masks, key=(lambda x: x["predicted_iou"]), reverse=False)
    return sorted_masks


def get_mask_rgb(masks):
    mask_rgb = np.full(
        (masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3),
        200,
        dtype=np.uint8,
    )

    for mask in masks:
        rgb = np.random.randint(0, 256, size=3, dtype=np.uint8)
        mask_rgb[mask["segmentation"]] = rgb

    return mask_rgb
