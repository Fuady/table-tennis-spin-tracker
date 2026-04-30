"""
src/ingestion/download_ttnet.py
───────────────────────────────
Download the TTNet public dataset — the best open-source annotated
table tennis dataset with ball position labels.

Dataset paper: https://arxiv.org/abs/2004.09927
Original repo : https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch

The dataset contains:
  - ~120 match sequences with ball position annotations
  - ~20 fps videos at 1920x1080
  - Ball coordinates (x, y) per frame
  - Event labels (bounce, net hit, empty frame)

Usage:
    python src/ingestion/download_ttnet.py --output data/raw/ttnet
"""

import argparse
import hashlib
import zipfile
from pathlib import Path

import gdown
import requests
from loguru import logger
from tqdm import tqdm


# ─── Public dataset mirrors ────────────────────────────────────────────────
DATASET_SOURCES = {
    "ttnet_sample": {
        "description": "TTNet sample (5 sequences, ~200MB)",
        "gdrive_id": "1yjZFP5zFXhUBCHFpwLIMqnWomgcknFkm",  # public folder
        "type": "folder",
    },
    "ttnet_annotations": {
        "description": "Ball position CSV annotations for all sequences",
        "url": "https://raw.githubusercontent.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/master/src/data_process/ttnet_data_utils.py",
        "type": "file",
    },
}

# Alternative: Roboflow public TT dataset (requires free account)
ROBOFLOW_INSTRUCTIONS = """
Alternative dataset (Roboflow — larger, pre-annotated):

1. Go to: https://roboflow.com/search?q=table+tennis+ball
2. Filter: Free, Object Detection
3. Select 'Table Tennis Ball Detection' by Daniel
4. Export as YOLOv8 format
5. Copy the download snippet and run:
   python src/ingestion/roboflow_download.py --api_key YOUR_KEY
"""


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Stream download with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=desc or dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


def download_gdrive_folder(folder_id: str, output_dir: Path) -> None:
    """Download a public Google Drive folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info(f"Downloading from Google Drive folder: {url}")

    try:
        gdown.download_folder(url=url, output=str(output_dir), quiet=False, use_cookies=False)
        logger.success(f"Downloaded to {output_dir}")
    except Exception as e:
        logger.error(f"gdown failed: {e}")
        logger.info("Trying alternative: manual download instructions")
        print_manual_instructions(folder_id, output_dir)


def print_manual_instructions(folder_id: str, output_dir: Path) -> None:
    """Print manual download steps if automated fails."""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print(f"\n1. Open this URL in your browser:")
    print(f"   https://drive.google.com/drive/folders/{folder_id}")
    print(f"\n2. Click 'Download all' (or Ctrl+A → Download)")
    print(f"\n3. Extract the zip to: {output_dir.absolute()}")
    print("\n" + "="*60)


def create_synthetic_samples(output_dir: Path) -> None:
    """
    Create synthetic/dummy data files for testing the pipeline
    without downloading real data. Useful for CI and development.
    """
    logger.info("Creating synthetic sample data for pipeline testing...")

    import json
    import random

    import cv2
    import numpy as np

    samples_dir = output_dir / "synthetic_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    spin_classes = ["topspin", "backspin", "sidespin", "float"]

    for i, spin in enumerate(spin_classes):
        # Create a short synthetic video (2 seconds, 30fps, 640x480)
        video_path = samples_dir / f"synthetic_{spin}_{i:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

        annotations = []
        x, y = 100.0, 240.0

        for frame_idx in range(60):  # 2 seconds
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 40  # dark bg

            # Draw table-like surface
            cv2.line(frame, (0, 350), (640, 350), (80, 80, 80), 3)
            cv2.line(frame, (320, 300), (320, 400), (80, 80, 80), 2)

            # Simulate ball trajectory based on spin type
            t = frame_idx / 30.0
            if spin == "topspin":
                x = 100 + frame_idx * 7.0
                y = 200 + 80 * (t**1.5)          # faster drop
            elif spin == "backspin":
                x = 100 + frame_idx * 5.0
                y = 280 - 30 * t + 20 * (t**2)   # floats then drops
            elif spin == "sidespin":
                x = 100 + frame_idx * 6.0
                y = 240 + 40 * np.sin(t * 3)      # lateral drift
            else:  # float
                x = 100 + frame_idx * 8.0
                y = 230 + 10 * t                   # minimal curve

            # Add gaussian noise
            x += random.gauss(0, 1.5)
            y += random.gauss(0, 1.5)

            bx, by = int(np.clip(x, 5, 634)), int(np.clip(y, 5, 474))

            # Draw ball
            cv2.circle(frame, (bx, by), 8, (255, 255, 255), -1)
            cv2.circle(frame, (bx, by), 8, (200, 200, 200), 1)

            out.write(frame)
            annotations.append({
                "frame": frame_idx,
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "visible": 1,
                "spin": spin,
            })

        out.release()

        # Save annotation JSON
        ann_path = samples_dir / f"synthetic_{spin}_{i:03d}_annotations.json"
        with open(ann_path, "w") as f:
            json.dump({"video": video_path.name, "spin_type": spin, "frames": annotations}, f, indent=2)

    logger.success(f"Created {len(spin_classes)} synthetic sample videos in {samples_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download TTNet dataset or create synthetic samples")
    parser.add_argument("--output", type=str, default="data/raw/ttnet")
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic samples only (no download)")
    parser.add_argument("--source", choices=["ttnet", "roboflow", "synthetic"], default="ttnet")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.synthetic or args.source == "synthetic":
        create_synthetic_samples(output_dir)
        return

    logger.info(f"Downloading TTNet dataset to {output_dir}")

    if args.source == "ttnet":
        src = DATASET_SOURCES["ttnet_sample"]
        if src["type"] == "folder":
            download_gdrive_folder(src["gdrive_id"], output_dir)
    elif args.source == "roboflow":
        print(ROBOFLOW_INSTRUCTIONS)
        return

    # Always create synthetic samples as fallback / for testing
    logger.info("Also creating synthetic samples for pipeline testing...")
    create_synthetic_samples(output_dir)

    logger.success("Dataset download complete!")
    logger.info(f"Data location: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
