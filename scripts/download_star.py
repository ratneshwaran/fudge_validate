"""Download the STAR dataset from GitHub."""
import subprocess
import sys
from pathlib import Path


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    star_dir = data_dir / "STAR"

    if star_dir.exists():
        print(f"STAR already exists at {star_dir}")
        return

    data_dir.mkdir(exist_ok=True)
    print(f"Cloning STAR dataset to {star_dir} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/RasaHQ/STAR.git", str(star_dir)],
        check=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
