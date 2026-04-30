"""
src/processing/extract_frames.py
─────────────────────────────────
Extract frames from raw videos at a target FPS.
Outputs:
  - JPEG frames in data/processed/frames/<video_id>/
  - A metadata CSV: data/processed/frame_index.csv

Usage:
    python src/processing/extract_frames.py \
        --input data/raw/ \
        --output data/processed/ \
        --fps 30 \
        --resize 640 640
"""

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_video_id(video_path: Path) -> str:
    """Generate a short stable ID from the filename."""
    h = hashlib.md5(video_path.name.encode()).hexdigest()[:8]
    stem = video_path.stem[:20].replace(" ", "_")
    return f"{stem}_{h}"


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: int = 30,
    resize: tuple[int, int] | None = (640, 640),
    max_frames: int | None = None,
) -> list[dict]:
    """
    Extract frames from a single video.

    Returns:
        List of frame metadata dicts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if src_fps <= 0:
        src_fps = 30.0

    # Frame sampling interval
    interval = max(1, int(round(src_fps / target_fps)))
    video_id = get_video_id(video_path)
    frames_dir = output_dir / "frames" / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    records = []
    frame_idx = 0
    saved_idx = 0

    logger.info(f"Extracting: {video_path.name} | {src_fps:.1f}fps → {target_fps}fps | {total_frames} frames")

    with tqdm(total=total_frames, desc=video_path.name[:40], unit="frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                if resize:
                    frame_out = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                else:
                    frame_out = frame

                frame_filename = f"frame_{saved_idx:06d}.jpg"
                frame_path = frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame_out, [cv2.IMWRITE_JPEG_QUALITY, 92])

                records.append({
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "frame_filename": frame_filename,
                    "frame_path": str(frame_path),
                    "original_frame_idx": frame_idx,
                    "saved_frame_idx": saved_idx,
                    "timestamp_sec": round(frame_idx / src_fps, 3),
                    "orig_width": width,
                    "orig_height": height,
                    "out_width": resize[0] if resize else width,
                    "out_height": resize[1] if resize else height,
                })
                saved_idx += 1

                if max_frames and saved_idx >= max_frames:
                    break

            frame_idx += 1
            pbar.update(1)

    cap.release()
    logger.success(f"  → Saved {saved_idx} frames to {frames_dir}")
    return records


def compute_frame_quality(frame_path: Path) -> float:
    """Compute Laplacian variance as sharpness proxy. Low = blurry."""
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def main():
    parser = argparse.ArgumentParser(description="Extract frames from raw table tennis videos")
    parser.add_argument("--input", type=str, default="data/raw", help="Root folder with videos")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resize", type=int, nargs=2, default=[640, 640], metavar=("W", "H"))
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--quality_filter", type=float, default=50.0,
                        help="Minimum sharpness score (Laplacian variance). 0 = no filter.")
    args = parser.parse_args()

    config = load_config(args.config)
    extensions = tuple(config["data"]["video_extensions"])

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = list(input_dir.rglob("*"))
    video_files = [v for v in video_files if v.suffix.lower() in extensions]

    if not video_files:
        logger.warning(f"No videos found in {input_dir}")
        logger.info("Try: python src/ingestion/download_ttnet.py --synthetic")
        return

    logger.info(f"Found {len(video_files)} videos")

    all_records = []
    for video_path in video_files:
        records = extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            target_fps=args.fps,
            resize=tuple(args.resize),
            max_frames=args.max_frames,
        )

        # Optional: filter blurry frames
        if args.quality_filter > 0:
            filtered = []
            for rec in records:
                q = compute_frame_quality(Path(rec["frame_path"]))
                rec["sharpness"] = round(q, 2)
                if q >= args.quality_filter:
                    filtered.append(rec)
            logger.info(f"  Quality filter: kept {len(filtered)}/{len(records)} frames (sharpness >= {args.quality_filter})")
            records = filtered

        all_records.extend(records)

    # Save CSV index
    index_path = output_dir / "frame_index.csv"
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        logger.success(f"Frame index saved → {index_path} ({len(all_records)} frames total)")
    else:
        logger.warning("No frames extracted.")


if __name__ == "__main__":
    main()
