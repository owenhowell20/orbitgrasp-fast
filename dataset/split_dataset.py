import os
import random
import shutil
from pathlib import Path
import argparse


def split_and_rename_dataset(camera_setting, root_dirs, output_dir, train_ratio=0.8, file_extension=".pkl"):
    """
    Splits dataset files into training and testing sets, renames them, and organizes into folders.

    Parameters:
    - root_dirs (list of str): List of root directories to search files in.
    - output_dir (str): Directory to save train/test sets.
    - train_ratio (float): Proportion of files to use for training (default 0.95).
    - file_extension (str): File extension to filter files (e.g., ".txt" or ".png").

    Returns:
    - None
    """
    root_dirs = [str(Path(root_dirs) / f"packed_{camera_setting}"), str(Path(root_dirs) / f"pile_{camera_setting}")]
    all_files = []
    for root_dir in root_dirs:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(file_extension):
                    all_files.append(os.path.join(root, file))

    if not all_files:
        print("No files found with the specified extension.")
        return

    random.seed(42)  # For reproducibility
    random.shuffle(all_files)

    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    train_dir = os.path.join(output_dir, f"train_{camera_setting}")
    test_dir = os.path.join(output_dir, f"test_{camera_setting}")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Helper function to copy and rename files
    def copy_and_rename(files, output_subdir, prefix):
        for idx, file_path in enumerate(files):
            new_name = f"{prefix}_scene_{(idx + 1) :05}{file_extension}"
            new_path = os.path.join(output_subdir, new_name)
            shutil.copy(file_path, new_path)

    copy_and_rename(train_files, train_dir, "train")
    copy_and_rename(test_files, test_dir, "test")

    print(f"Dataset split completed!")
    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--camera_setting', type=str, default='single')
    parser.add_argument(
        '--root_dir', type=str,
        default=str(Path(__file__).resolve().parent / "collected_data/se3_filtered"),
    )
    parser.add_argument('--output_dir', type=str,
                        default=str(Path(__file__).resolve().parent / "collected_data/se3_filtered"))
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()

    root_dir = args.root_dir
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    camera_setting = args.camera_setting

    split_and_rename_dataset(camera_setting, root_dir, output_dir, train_ratio=train_ratio, file_extension=".pkl")
