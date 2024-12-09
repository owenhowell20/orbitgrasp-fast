import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from se3_grasper_bce import OrbitGrasper
from utils.torch_utils import set_seed
from dataloader_dynamicR_wo_crop import orbitgrasp_dataset


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def train(config):
    # Set seed
    set_seed(config['seed'])

    # Create train and test datasets
    train_dataset = orbitgrasp_dataset(
        path=config['train_dataset']['path'],
        lmax=config['train_dataset']['lmax'],
        augment=False,
        load_harmonics=config['train_dataset']['load_harmonics'],
    )

    test_dataset = orbitgrasp_dataset(
        path=config['test_dataset']['path'],
        lmax=config['test_dataset']['lmax'],
        augment=config['test_dataset']['augment'],
        load_harmonics=config['test_dataset']['load_harmonics'],
        load_harmony_path=config['test_dataset']['load_harmony_path']
    )

    # Initialize OrbitGrasper
    orbit_grasper = OrbitGrasper(
        device=config['orbit_grasper']['device'],
        lr=config['orbit_grasper']['lr'],
        load=config['orbit_grasper']['load'],
        param_dir=config['orbit_grasper']['param_dir'],
        num_channel=config['orbit_grasper']['num_channel'],
        lmax=config['orbit_grasper']['lmax'],
        mmax=config['orbit_grasper']['mmax'],
        training_config=config
    )

    # Training

    orbit_grasper.train_test_save_aug(
        train_dataset,
        test_dataset=test_dataset,
        save_interval=config['training']['save_interval'],
        log=config['training']['log'],
        tr_epoch=config['training']['tr_epoch'],
        balance=config['training']['balance']
    )


if __name__ == '__main__':
    # Load the config
    config_path = "./scripts/single_config.yaml"
    # config_path = "./training_config.yaml"
    config = load_config(config_path)

    # Convert relative paths to absolute paths
    train_path = './dataset/collected_data/se3_filtered_random_back/train'
    test_path = './dataset/collected_data/se3_filtered_random_back/test'
    load_harmony_path = "./scripts/grasp_harmonics_l3_test_single.pt"

    # Update paths in the config
    config['train_dataset']['path'] = str(train_path)
    config['test_dataset']['path'] = str(test_path)
    config['test_dataset']['load_harmony_path'] = str(load_harmony_path)

    # Run training
    train(config)
