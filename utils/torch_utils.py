import os
import random
import numpy as np
import torch


def set_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def create_csv(path, columns):
    with open(path, "w") as f:
        f.write(",".join(columns))
        f.write("\n")


def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with open(path, "a") as f:
        f.write(row)
        f.write("\n")


def write_training(root, epoch, step, loss):
    # TODO concurrent writes could be an issue
    csv_path = os.path.join(root, "training_loss.csv")
    if not os.path.exists(csv_path):
        create_csv(
            csv_path,
            ["epoch", "step", "loss"],
        )
    append_csv(csv_path, epoch, step, loss)


def write_test(
    root,
    success_grasp,
    total_grasp,
    total_objs,
    grasped_objs,
    remain_objs,
    success_rate,
    declutter_rate,
    scene="packed",
):
    csv_path = os.path.join(root, f"{scene}_test_acc.csv")
    if not os.path.exists(csv_path):
        create_csv(
            csv_path,
            [
                "success_grasp",
                "total_grasp",
                "total_objs",
                "grasped_objs",
                "remain_objs",
                "success_rate",
                "declutter_rate",
            ],
        )
    append_csv(
        csv_path,
        success_grasp,
        total_grasp,
        total_objs,
        grasped_objs,
        remain_objs,
        success_rate,
        declutter_rate,
    )


def write_log(root, test_loss, success_rate, max_success_rate, scene="packed"):
    csv_path = os.path.join(root, f"{scene}_test_log.csv")
    if not os.path.exists(csv_path):
        create_csv(
            csv_path,
            ["test_loss", "success_rate"],
        )
    append_csv(csv_path, test_loss, success_rate)
