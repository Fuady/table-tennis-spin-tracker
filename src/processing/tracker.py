"""
src/processing/tracker.py
──────────────────────────
Ball detection (YOLOv8) + multi-object tracking (DeepSORT).
Extracts per-rally trajectory sequences and saves them as
structured data for the spin classifier.

Usage:
    python src/processing/tracker.py \
        --video data/raw/sample_rally.mp4 \
        --output data/processed/trajectories/ \
        --visualize
"""

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. Detection will be simulated.")

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logger.warning("deep_sort_realtime not installed. Tracking will use centroid fallback.")


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class BallDetection:
    frame_idx: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    track_id: Optional[int] = None


@dataclass
class Trajectory:
    track_id: int
    detections: list[BallDetection] = field(default_factory=list)
    spin_label: Optional[str] = None  # filled after annotation

    @property
    def length(self) -> int:
        return len(self.detections)

    @property
    def positions(self) -> np.ndarray:
        """Nx2 array of (x, y) center positions."""
        return np.array([[d.x_center, d.y_center] for d in self.detections])

    @property
    def velocities(self) -> np.ndarray:
        """(N-1)x2 array of frame-to-frame velocity vectors."""
        pos = self.positions
        return np.diff(pos, axis=0)

    @property
    def speeds(self) -> np.ndarray:
        """(N-1) array of pixel-per-frame speeds."""
        return np.linalg.norm(self.velocities, axis=1)

    def compute_curvature(self) -> float:
        """Mean curvature of the trajectory (deviation from straight line)."""
        pos = self.positions
        if len(pos) < 3:
            return 0.0
        # Fit a straight line and compute mean residual
        t = np.arange(len(pos))
        x_fit = np.polyfit(t, pos[:, 0], 1)
        y_fit = np.polyfit(t, pos[:, 1], 1)
        x_resid = pos[:, 0] - np.polyval(x_fit, t)
        y_resid = pos[:, 1] - np.polyval(y_fit, t)
        return float(np.mean(np.sqrt(x_resid**2 + y_resid**2)))

    def compute_lateral_drift(self) -> float:
        """Max lateral deviation — indicator of side spin."""
        pos = self.positions
        if len(pos) < 2:
            return 0.0
        # Project positions onto line from first to last point
        v = pos[-1] - pos[0]
        v_norm = v / (np.linalg.norm(v) + 1e-6)
        deviations = []
        for p in pos:
            diff = p - pos[0]
            proj = np.dot(diff, v_norm) * v_norm
            lat = np.linalg.norm(diff - proj)
            deviations.append(lat)
        return float(np.max(deviations))

    def compute_arc_direction(self) -> str:
        """
        Classify vertical arc direction.
        - 'concave_down': ball dips quickly → topspin signature
        - 'concave_up': ball floats → backspin signature
        - 'neutral': minimal vertical curvature
        """
        pos = self.positions
        if len(pos) < 5:
            return "neutral"
        y = pos[:, 1]
        t = np.arange(len(y))
        coeffs = np.polyfit(t, y, 2)
        # coeffs[0] is the 2nd-degree term (concavity)
        if coeffs[0] > 0.05:
            return "concave_down"   # y increases → drops faster (topspin in image coords)
        elif coeffs[0] < -0.05:
            return "concave_up"     # y decreases → floats up (backspin)
        return "neutral"

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "length": self.length,
            "spin_label": self.spin_label,
            "positions": self.positions.tolist(),
            "mean_speed": float(np.mean(self.speeds)) if len(self.speeds) > 0 else 0.0,
            "max_speed": float(np.max(self.speeds)) if len(self.speeds) > 0 else 0.0,
            "curvature": self.compute_curvature(),
            "lateral_drift": self.compute_lateral_drift(),
            "arc_direction": self.compute_arc_direction(),
            "detections": [asdict(d) for d in self.detections],
        }


class BallTracker:
    """
    Wraps YOLOv8 detection + DeepSORT tracking.
    Falls back to a simple IoU centroid tracker when deps are missing.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.det_cfg = config["detector"]
        self.trk_cfg = config["tracker"]
        self._setup_detector()
        self._setup_tracker()

    def _setup_detector(self):
        if YOLO_AVAILABLE:
            model_path = self.det_cfg.get("fine_tuned_weights")
            if model_path and Path(model_path).exists():
                logger.info(f"Loading fine-tuned detector: {model_path}")
                self.detector = YOLO(model_path)
            else:
                logger.info(f"Loading base YOLO model: {self.det_cfg['model']}")
                self.detector = YOLO(self.det_cfg["model"])
        else:
            self.detector = None

    def _setup_tracker(self):
        if DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(
                max_age=self.trk_cfg["max_age"],
                n_init=self.trk_cfg["min_hits"],
                max_iou_distance=self.trk_cfg["iou_threshold"],
            )
        else:
            self.tracker = None
            self._simple_tracks: dict[int, list] = {}
            self._next_id = 0

    def detect(self, frame: np.ndarray) -> list[BallDetection]:
        """Run YOLOv8 on a single frame, return list of detections."""
        if self.detector is None:
            return self._simulate_detection(frame)

        h, w = frame.shape[:2]
        results = self.detector(
            frame,
            conf=self.det_cfg["confidence_threshold"],
            iou=self.det_cfg["iou_threshold"],
            classes=self.det_cfg.get("classes"),
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1
                detections.append(BallDetection(
                    frame_idx=0,
                    x_center=cx / w,  # normalize 0-1
                    y_center=cy / h,
                    width=bw / w,
                    height=bh / h,
                    confidence=conf,
                ))
        return detections

    def _simulate_detection(self, frame: np.ndarray) -> list[BallDetection]:
        """Fallback: detect white circular blobs (heuristic for TT ball)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        # Detect white blobs
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 2000:  # reasonable ball size
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
                if circularity > 0.6:  # must be round-ish
                    detections.append(BallDetection(
                        frame_idx=0,
                        x_center=x / w,
                        y_center=y / h,
                        width=(2 * radius) / w,
                        height=(2 * radius) / h,
                        confidence=float(circularity),
                    ))
        return detections[:1]  # keep only the best candidate

    def update(self, detections: list[BallDetection], frame: np.ndarray) -> list[BallDetection]:
        """Update tracker with new detections. Returns detections with track IDs."""
        if not detections:
            return []

        h, w = frame.shape[:2]

        if self.tracker is not None:
            # DeepSORT format: [[x1, y1, w, h, conf], ...]
            det_list = []
            for d in detections:
                x1 = (d.x_center - d.width / 2) * w
                y1 = (d.y_center - d.height / 2) * h
                bw = d.width * w
                bh = d.height * h
                det_list.append(([x1, y1, bw, bh], d.confidence, "ball"))

            tracks = self.tracker.update_tracks(det_list, frame=frame)
            result = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                ltrb = track.to_ltrb()
                cx = ((ltrb[0] + ltrb[2]) / 2) / w
                cy = ((ltrb[1] + ltrb[3]) / 2) / h
                matched = min(detections, key=lambda d: abs(d.x_center - cx) + abs(d.y_center - cy))
                matched.track_id = track.track_id
                result.append(matched)
            return result
        else:
            # Simple fallback: assign incrementing IDs
            for d in detections:
                d.track_id = 0  # single-object assumption
            return detections


def process_video(
    video_path: Path,
    output_dir: Path,
    config: dict,
    visualize: bool = False,
    min_trajectory_length: int = 10,
) -> list[Trajectory]:
    """
    Full pipeline: load video → detect → track → extract trajectories.

    Returns list of Trajectory objects (one per rally segment).
    """
    tracker = BallTracker(config)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Track active trajectories
    active: dict[int, Trajectory] = {}
    finished: list[Trajectory] = []

    if visualize:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vis_path = output_dir / f"{video_path.stem}_tracked.mp4"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vis_writer = cv2.VideoWriter(str(vis_path), fourcc, fps, (w, h))
        trail: dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
    else:
        vis_writer = None

    COLORS = [(0, 255, 0), (255, 165, 0), (0, 165, 255), (255, 0, 255)]

    logger.info(f"Processing: {video_path.name} ({total} frames @ {fps:.1f}fps)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = tracker.detect(frame)
        for d in detections:
            d.frame_idx = frame_idx

        tracked = tracker.update(detections, frame)

        seen_ids = set()
        for d in tracked:
            tid = d.track_id
            seen_ids.add(tid)
            if tid not in active:
                active[tid] = Trajectory(track_id=tid)
            active[tid].detections.append(d)

            if visualize and tid is not None:
                h_f, w_f = frame.shape[:2]
                px = int(d.x_center * w_f)
                py = int(d.y_center * h_f)
                trail[tid].append((px, py))
                color = COLORS[tid % len(COLORS)]
                # Draw trail
                pts = list(trail[tid])
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    t_color = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, pts[i - 1], pts[i], t_color, 2)
                cv2.circle(frame, (px, py), 8, color, -1)
                cv2.putText(frame, f"ID:{tid}", (px + 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Collect finished tracks (not seen this frame)
        for tid in list(active.keys()):
            if tid not in seen_ids:
                traj = active.pop(tid)
                if traj.length >= min_trajectory_length:
                    finished.append(traj)

        if visualize and vis_writer:
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            vis_writer.write(frame)

        frame_idx += 1

    # Collect remaining active tracks
    for traj in active.values():
        if traj.length >= min_trajectory_length:
            finished.append(traj)

    cap.release()
    if vis_writer:
        vis_writer.release()
        logger.info(f"Visualization saved → {vis_path}")

    logger.success(f"Extracted {len(finished)} trajectories from {video_path.name}")

    # Save trajectories to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_trajectories.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "video": str(video_path),
                "fps": fps,
                "total_frames": total,
                "trajectories": [t.to_dict() for t in finished],
            },
            f,
            indent=2,
        )
    logger.info(f"Trajectories saved → {out_path}")

    return finished


def main():
    parser = argparse.ArgumentParser(description="Track table tennis ball and extract trajectories")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/processed/trajectories")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--min_len", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    process_video(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        config=config,
        visualize=args.visualize,
        min_trajectory_length=args.min_len,
    )


if __name__ == "__main__":
    main()
