import os
import requests
import zipfile

def download_ettm1(data_dir="data"):
    url = "https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm1.csv"
    target_dir = os.path.join(data_dir, "ETT-small")
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, "ETTm1.csv")

    if os.path.exists(target_file):
        print(f" Already downloaded: {target_file}")
        return

    print(f" Downloading ETTm1.csv to {target_file}...")
    r = requests.get(url)
    with open(target_file, "wb") as f:
        f.write(r.content)

    print(" Download complete!")

if __name__ == "__main__":
    download_ettm1()
