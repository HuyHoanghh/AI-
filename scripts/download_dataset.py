from pathlib import Path
from urllib.request import urlretrieve

from config.settings import PROJECT_ROOT

BASE_URL = "https://raw.githubusercontent.com/trannguyenthaituan251209/vietnamese_sms_dataset/main"
FILES = ["train.csv", "test.csv"]


def main():
    target_dir = PROJECT_ROOT / "data" / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        target = target_dir / filename
        if target.exists():
            print(f"Skip: {target} already exists")
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {url}")
        urlretrieve(url, target)
        print(f"Saved: {target}")

    print("Dataset ready.")


if __name__ == "__main__":
    main()
