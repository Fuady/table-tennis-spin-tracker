"""
src/ingestion/youtube_scraper.py
─────────────────────────────────
Download table tennis videos from YouTube using yt-dlp.

Usage:
    python src/ingestion/youtube_scraper.py \
        --query "table tennis topspin slow motion" \
        --max_videos 20 \
        --output data/raw/youtube

Dependencies: yt-dlp (pip install yt-dlp)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def search_and_download(
    query: str,
    output_dir: Path,
    max_videos: int = 10,
    min_duration: int = 30,
    max_duration: int = 600,
    quality: str = "720",
) -> list[Path]:
    """
    Search YouTube for a query and download matching videos.

    Args:
        query: Search string
        output_dir: Folder to save videos
        max_videos: Maximum number to download
        min_duration: Minimum video length in seconds
        max_duration: Maximum video length in seconds
        quality: Video quality (360, 480, 720, 1080)

    Returns:
        List of downloaded file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # yt-dlp format selector: best video+audio up to target quality
    format_str = f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality}][ext=mp4]/best"

    output_template = str(output_dir / "%(title)s_%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        f"ytsearch{max_videos}:{query}",
        "--format", format_str,
        "--output", output_template,
        "--merge-output-format", "mp4",
        "--match-filter", f"duration > {min_duration} & duration < {max_duration}",
        "--write-info-json",
        "--no-playlist",
        "--ignore-errors",
        "--quiet",
        "--progress",
        "--no-warnings",
    ]

    logger.info(f"Searching YouTube: '{query}' (max {max_videos} videos)")
    logger.info(f"Duration filter: {min_duration}s – {max_duration}s")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode not in (0, 1):  # 1 = partial success (some skipped)
            logger.warning(f"yt-dlp exited with code {result.returncode}")
    except FileNotFoundError:
        logger.error("yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)

    downloaded = list(output_dir.glob("*.mp4"))
    logger.success(f"Downloaded {len(downloaded)} videos to {output_dir}")
    return downloaded


def save_download_manifest(output_dir: Path, queries: list[str]) -> None:
    """Save a JSON manifest of all downloaded videos with metadata."""
    manifest = []
    for json_file in output_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                info = json.load(f)
            manifest.append({
                "id": info.get("id"),
                "title": info.get("title"),
                "duration": info.get("duration"),
                "url": info.get("webpage_url"),
                "uploader": info.get("uploader"),
                "view_count": info.get("view_count"),
                "upload_date": info.get("upload_date"),
                "filename": info.get("_filename"),
                "query": next((q for q in queries if q in json_file.name), "unknown"),
            })
        except Exception as e:
            logger.warning(f"Could not parse {json_file.name}: {e}")

    manifest_path = output_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved → {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download table tennis videos from YouTube")
    parser.add_argument("--query", type=str, help="Search query (overrides config)")
    parser.add_argument("--max_videos", type=int, default=10)
    parser.add_argument("--output", type=str, default="data/raw/youtube")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--quality", type=str, default="720")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["ingestion"]["youtube"]

    output_dir = Path(args.output)
    queries = [args.query] if args.query else cfg["queries"]

    all_downloaded = []
    for q in queries:
        downloaded = search_and_download(
            query=q,
            output_dir=output_dir / q.replace(" ", "_")[:40],
            max_videos=args.max_videos,
            min_duration=cfg["min_duration_sec"],
            max_duration=cfg["max_duration_sec"],
            quality=args.quality,
        )
        all_downloaded.extend(downloaded)

    save_download_manifest(output_dir, queries)
    logger.success(f"Total downloaded: {len(all_downloaded)} videos")


if __name__ == "__main__":
    main()
