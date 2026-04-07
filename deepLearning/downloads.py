import os
import urllib.request
import tarfile
from pathlib import Path

def download_dataset(target_dir="flower_data"):
    """
    Download and unpack Oxford 102 Flower dataset into `target_dir`.
    Files created:
      - flower_data/102flowers.tgz
      - flower_data/imagelabels.mat
      - flower_data/jpg/...
    """

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"

    archive_path = target_dir / "102flowers.tgz"
    labels_path = target_dir / "imagelabels.mat"
    extract_dir = target_dir / "jpg"

    if not archive_path.exists():
        print(f"Downloading dataset archive to {archive_path} ...")
        urllib.request.urlretrieve(url, archive_path)
        print("Archive download complete.")
    else:
        print("Dataset archive already exists; skipping download.")

    if not labels_path.exists():
        print(f"Downloading labels to {labels_path} ...")
        urllib.request.urlretrieve(labels_url, labels_path)
        print("Labels download complete.")
    else:
        print("Labels already exist; skipping download.")

    if not extract_dir.exists():
        print(f"Extracting archive into {target_dir} ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            def safe_extract(tar_obj, path="."):
                for member in tar_obj.getmembers():
                    member_path = Path(path) / member.name
                    if not Path(path).resolve() in member_path.resolve().parents and member_path.resolve() != Path(path).resolve():
                        raise Exception("Attempted Path Traversal in Tar File")
                tar_obj.extractall(path=path)
            safe_extract(tar, path=str(target_dir))
        print("Extraction complete.")
    else:
        print("Extracted folder already exists; skipping extraction.")

if __name__ == "__main__":
    download_dataset("flower_data")