import os
import urllib.request

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
PRETRAINED_DIR = "pretrained"
SAM_FILENAME = "sam_vit_h_4b8939.pth"
SAM_PATH = os.path.join(PRETRAINED_DIR, SAM_FILENAME)


def main():
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    if os.path.exists(SAM_PATH):
        print(f"Checkpoint already exists at {SAM_PATH}")
    else:
        print(f"Downloading SAM ViT-H checkpoint to {SAM_PATH}...")
        urllib.request.urlretrieve(SAM_URL, SAM_PATH)
        print("Download complete.")


if __name__ == "__main__":
    main()
