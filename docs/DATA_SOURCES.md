# Data Sources & Collection Guide

## Overview

This project uses four types of data sources, in order of annotation quality:

| Source | Quality | Size | Effort | Best For |
|--------|---------|------|--------|----------|
| TTNet Dataset | High | ~8GB | Low (pre-labeled) | Supervised training |
| Self-recorded | Highest | Variable | High | Ground truth validation |
| YouTube (yt-dlp) | Medium | Variable | Low | Pre-training / augmentation |
| Roboflow | High | ~2GB | Medium | Detection fine-tuning |

---

## 1. TTNet Dataset (Recommended Start)

**Paper**: [TTNet: Real-time temporal and spatial video analysis of table tennis (CVPR 2020)](https://arxiv.org/abs/2004.09927)

**What it contains**:
- 120 match sequences (professional level)
- Ball (x, y) position per frame
- Event labels: `bounce`, `net`, `empty`
- ~20 fps, 1920×1080

**How to download**:
```bash
python src/ingestion/download_ttnet.py --source ttnet --output data/raw/ttnet
```

If the automated download fails (Google Drive rate limits), manually download from:
- https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch

**Important**: TTNet provides ball *position* labels but NOT spin labels. You must add spin labels manually (see Section 5).

---

## 2. YouTube Video Collection

**Tool**: [yt-dlp](https://github.com/yt-dlp/yt-dlp) — the most reliable YouTube downloader.

**Recommended search queries** (in `configs/config.yaml`):
```
- "table tennis topspin training slow motion"
- "table tennis backspin chop rally"
- "table tennis sidespin serve"
- "ping pong match broadcast full hd"
- "table tennis robot training"
```

**How to scrape**:
```bash
python src/ingestion/youtube_scraper.py \
    --query "table tennis topspin slow motion" \
    --max_videos 20 \
    --output data/raw/youtube/
```

**Good channels for data**:
- PingSkills (Australia) — excellent tutorial content with clear spin demonstration
- EmRatThich (France) — breakdown of Chinese national team techniques
- ITTF (official) — broadcast matches with slow-motion replays
- Tom Lodziak — beginner/intermediate training content

**Legal note**: Downloaded videos are for **research/non-commercial use only**. Do not redistribute. If deploying commercially, use only openly licensed or self-recorded data.

---

## 3. Self-Recorded Data (Best Quality)

For the most accurate spin labels, record your own videos.

**Setup requirements**:
- **Camera**: Any 60fps+ camera (phone works). Higher frame rate = better motion capture.
- **Background**: Solid dark background — improves ball detection dramatically.
- **Lighting**: 3-point lighting, no flickering. Avoid strong shadows behind the ball.
- **Mounting**: Fixed tripod at table height, ~3 meters back from the table end.

**Recording protocol**:
1. Set up camera with clear view of the full table
2. Have a player execute 20+ serves of each spin type
3. Announce spin type before each series: *"topspin series starting"*
4. Record at 60fps minimum, 120fps preferred for spin classification

**Annotation**:
- Use the timestamp from the announce to label spin type per rally
- Fine-tune with `scripts/annotate.py` for frame-level labels

---

## 4. Roboflow Public Dataset

Roboflow hosts many table tennis object detection datasets with YOLO-format labels (ball bounding boxes).

**Best datasets**:
1. Search: https://roboflow.com/search?q=table+tennis+ball
2. Filter: Free, Object Detection
3. Recommended: "Table Tennis Ball Detection" — ~3,000 labeled frames

**Download**:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")  # Free key at roboflow.com
project = rf.workspace().project("table-tennis-ball-detection")
dataset = project.version(1).download("yolov8")
```

---

## 5. Spin Labeling Guide

TTNet and YouTube data need spin labels added manually.

**Criteria for labeling** (watch in 0.25x slow motion):

| Spin | Visual Cue | Trajectory | Player Stroke |
|------|-----------|------------|---------------|
| **Topspin** | Ball dips sharply, bounces forward | Strong downward arc | Brush upward over ball |
| **Backspin** | Ball floats, bounces back | Flat or slight upward float | Brush downward under ball |
| **Sidespin** | Ball curves left or right | Lateral deviation > straight line | Brush across ball sideways |
| **Float** | No spin, unpredictable | Nearly straight | Flat push/hit |

**Semi-automatic annotation** (after initial detection):
```bash
python scripts/annotate.py \
    --video data/raw/my_video.mp4 \
    --trajectories data/processed/trajectories/my_video_trajectories.json
```

---

## 6. Data Quality Checklist

Before using any video in training:

- [ ] Ball is fully visible for at least 10 consecutive frames per rally
- [ ] Video is at least 30fps (60fps preferred)
- [ ] Ball is not occluded by paddle, net, or player for >3 consecutive frames
- [ ] Lighting is consistent (no flicker)
- [ ] Spin type is confidently identifiable in slow motion
- [ ] Video is not artificially sped up or slowed down

---

## 7. Dataset Statistics (Target)

For a well-performing model, aim for:

| Spin Type | Minimum | Recommended |
|-----------|---------|-------------|
| Topspin | 200 | 500+ |
| Backspin | 200 | 500+ |
| Sidespin | 150 | 300+ |
| Float | 100 | 200+ |
| **Total** | **650** | **1500+** |

Current realistic expectation from TTNet + 20 YouTube videos: ~400-800 labeled trajectories.
