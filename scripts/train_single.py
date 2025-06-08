import sys
import os
import yaml
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from se3_grasper_bce import OrbitGrasper
from utils.torch_utils import set_seed
from dataloader_dynamicR import orbitgrasp_dataset


def find_checkpoint(root_dir, prefix):
    files = os.listdir(root_dir)
    matching_files = [f for f in files if f"-ckpt-14-" in f and f.endswith(".pt")]
    if not matching_files:
        raise ValueError(f"No checkpoints found with prefix '{prefix}' in '{root_dir}'")
    # Sort by filename to find the latest one
    matching_files.sort(reverse=True)
    return matching_files[0]


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train(config):
    # Set seed
    set_seed(config["seed"])

    # Create train and test datasets
    train_dataset = orbitgrasp_dataset(
        path=config["train_dataset"]["path"],
        lmax=config["train_dataset"]["lmax"],
        augment=config["train_dataset"]["augment"],
        load_harmonics=config["train_dataset"]["load_harmonics"],
        augment_ratio=config["train_dataset"]["augment_ratio"],
        min_allowed_points=config["train_dataset"]["min_allowed_points"],
        max_allowed_points=config["train_dataset"]["max_allowed_points"],
    )

    test_dataset = orbitgrasp_dataset(
        path=config["test_dataset"]["path"],
        lmax=config["test_dataset"]["lmax"],
        augment=config["test_dataset"]["augment"],
        load_harmonics=config["test_dataset"]["load_harmonics"],
        min_allowed_points=config["test_dataset"]["min_allowed_points"],
        max_allowed_points=config["test_dataset"]["max_allowed_points"],
        load_harmony_path=config["test_dataset"]["load_harmony_path"],
    )

    load_name = None
    if config["orbit_grasper"]["load"] != 0:
        load_name = find_checkpoint(
            config["orbit_grasper"]["param_dir"], config["orbit_grasper"]["load"]
        )

    # Initialize OrbitGrasper
    orbit_grasper = OrbitGrasper(
        device=config["orbit_grasper"]["device"],
        lr=config["orbit_grasper"]["lr"],
        load=config["orbit_grasper"]["load"],
        load_name=load_name,
        param_dir=config["orbit_grasper"]["param_dir"],
        num_channel=config["orbit_grasper"]["num_channel"],
        lmax=config["orbit_grasper"]["lmax"],
        mmax=config["orbit_grasper"]["mmax"],
        training_config=config,
    )

    # Training

    orbit_grasper.train_test_save_aug(
        train_dataset,
        test_dataset=test_dataset,
        save_interval=config["training"]["save_interval"],
        log=config["training"]["log"],
        tr_epoch=config["training"]["tr_epoch"],
        balance=config["training"]["balance"],
    )


if __name__ == "__main__":
    # Load the config
    config_path = Path(__file__).resolve().parent / "single_config.yaml"
    config = load_config(config_path)

    base_path = Path(__file__).resolve().parent.parent
    config["train_dataset"]["path"] = base_path / config["train_dataset"]["path"]
    config["test_dataset"]["path"] = base_path / config["test_dataset"]["path"]
    config["test_dataset"]["load_harmony_path"] = (
        base_path / config["test_dataset"]["load_harmony_path"]
    )
    config["orbit_grasper"]["param_dir"] = (
        base_path / config["orbit_grasper"]["param_dir"]
    )

    # Run training
    train(config)
