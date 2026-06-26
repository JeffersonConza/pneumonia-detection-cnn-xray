import os
import requests
import tarfile
from tqdm import tqdm

DATA_URL = "https://dsserver-prod-resources-1.s3.amazonaws.com/cnn/xray_dataset.tar.gz"

# Detect if running in Google Colab to choose download directory
import sys
IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')

if IS_COLAB:
    DATA_DIR = '/content/data'
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

TAR_PATH = os.path.join(DATA_DIR, 'xray_dataset.tar.gz')


def download_file(url, filename):
    # Perform a HEAD request to check the expected file size
    try:
        head_response = requests.head(url, timeout=10)
        expected_size = int(head_response.headers.get('content-length', 0))
    except Exception as e:
        print(f"Warning: HEAD request failed: {e}")
        expected_size = 0

    if os.path.exists(filename):
        if expected_size > 0:
            actual_size = os.path.getsize(filename)
            if actual_size == expected_size:
                print(f"File {filename} already exists and is fully downloaded. Skipping download.")
                return
            else:
                print(f"File {filename} is corrupted or incomplete ({actual_size} bytes vs expected {expected_size} bytes). Deleting and re-downloading...")
                try:
                    os.remove(filename)
                except Exception as e:
                    print(f"Error removing corrupted file: {e}")
        else:
            # Fallback if HEAD request was unsuccessful
            print(f"File {filename} already exists. Skipping download.")
            return

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 MB chunks

    with open(filename, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            ascii=True,
            ncols=100,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)
    print("Download complete.")



def extract_tar_gz(file_path, output_path):
    print(f"Extracting {file_path}...")
    with tarfile.open(file_path, 'r:gz') as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc="Extracting", unit="files") as bar:
            for member in members:
                tar.extract(member, path=output_path)
                bar.update(1)
    print("Extraction complete.")


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Download
    download_file(DATA_URL, TAR_PATH)

    # 2. Extract
    # This will extract the 'chest_xray' folder into 'data/'
    extract_tar_gz(TAR_PATH, DATA_DIR)
