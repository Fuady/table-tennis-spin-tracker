"""
src/api/main.py
──────────────────────────────────────
FastAPI production server for the spin detection system.

Endpoints:
  POST  /analyze          → upload video → get spin analysis JSON
  POST  /analyze/frame    → single frame → immediate prediction
  WS    /stream           → real-time webcam streaming
  GET   /health           → health check
  GET   /docs             → Swagger UI (auto-generated)

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import io
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from src.modeling.model import SPIN_CLASSES, build_model
from src.modeling.train_classifier import SCALAR_FEATURE_COLS
from src.processing.feature_engineering import build_feature_vector
from src.processing.tracker import BallTracker, process_video


# ─── Config & model loading ───────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class ModelRegistry:
    """Singleton to hold loaded model + config in memory."""
    _instance = None
    model = None
    config = None
    device = None
    scaler = None
    is_ready = False

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        clf_cfg = self.config["spin_classifier"]
        weights_path = Path(clf_cfg["weights_path"])

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if weights_path.exists():
            ckpt = torch.load(weights_path, map_location=self.device)
            scalar_dim = ckpt["scalar_feature_dim"]
            self.model = build_model(
                model_type=clf_cfg["model_type"],
                scalar_feature_dim=scalar_dim,
                lstm_hidden=clf_cfg["lstm_hidden"],
                lstm_layers=clf_cfg["lstm_layers"],
                dropout=0.0,
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self.is_ready = True
            logger.success(f"Model loaded from {weights_path}")
        else:
            logger.warning(f"No model weights at {weights_path}. API will return mock predictions.")
            self.is_ready = False


registry = ModelRegistry.get()


# ─── FastAPI app ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Table Tennis Spin Detector API",
    description="Real-time ball trajectory tracking and spin classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    registry.load()
    logger.info("API started")


# ─── Schemas ──────────────────────────────────────────────────────────────

class SpinResult(BaseModel):
    rally_id: int
    frame_start: int
    frame_end: int
    spin_type: str
    confidence: float
    all_probs: dict[str, float]
    trajectory_points: list[list[float]]
    ball_speed_px_per_frame: float
    arc_direction: str
    lateral_drift: float


class AnalysisResponse(BaseModel):
    job_id: str
    video_filename: str
    processing_time_sec: float
    total_rallies: int
    spin_summary: dict[str, int]
    results: list[SpinResult]
    model_ready: bool


# ─── Inference helper ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_spin(traj_dict: dict) -> tuple[str, float, dict[str, float]]:
    """Run spin classification on a trajectory dict."""
    reg = registry

    feats_dict = build_feature_vector(traj_dict, fps=30.0)
    if feats_dict is None:
        return "unknown", 0.0, {}

    if not reg.is_ready:
        # Mock prediction for demo without trained model
        probs = np.random.dirichlet([1, 1, 1, 1])
        spin = SPIN_CLASSES[np.argmax(probs)]
        return spin, float(np.max(probs)), {s: round(float(p), 4) for s, p in zip(SPIN_CLASSES, probs)}

    # Build scalar feature vector
    scalar_feats = np.array(
        [feats_dict.get(col, 0.0) for col in SCALAR_FEATURE_COLS],
        dtype=np.float32,
    )

    # Build trajectory tensor
    norm_traj = np.array(feats_dict["norm_traj"], dtype=np.float32)
    traj_t = torch.tensor(norm_traj, dtype=torch.float32).unsqueeze(0).to(reg.device)
    feat_t = torch.tensor(scalar_feats, dtype=torch.float32).unsqueeze(0).to(reg.device)

    logits = reg.model(traj_t, feat_t)
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return (
        SPIN_CLASSES[pred_idx],
        round(float(probs[pred_idx]), 4),
        {s: round(float(p), 4) for s, p in zip(SPIN_CLASSES, probs)},
    )


# ─── Routes ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_ready": registry.is_ready,
        "device": str(registry.device) if registry.device else "not loaded",
        "timestamp": time.time(),
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(video: UploadFile = File(...)):
    """
    Upload a video file and receive full spin analysis.
    Supports: mp4, avi, mov
    """
    cfg = registry.config or load_config()
    allowed = cfg["api"]["allowed_extensions"]

    suffix = Path(video.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported format: {suffix}. Allowed: {allowed}")

    max_size = cfg["api"]["max_video_size_mb"] * 1024 * 1024
    content = await video.read()
    if len(content) > max_size:
        raise HTTPException(413, f"File too large. Max {cfg['api']['max_video_size_mb']}MB")

    job_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        out_dir = Path(f"/tmp/tt_tracker_{job_id}")
        trajectories = process_video(
            video_path=tmp_path,
            output_dir=out_dir,
            config=cfg,
            visualize=False,
            min_trajectory_length=cfg["data"]["min_trajectory_length"],
        )

        results = []
        spin_summary = {s: 0 for s in SPIN_CLASSES}

        for i, traj in enumerate(trajectories):
            traj_dict = traj.to_dict()
            spin_type, confidence, all_probs = predict_spin(traj_dict)
            spin_summary[spin_type] = spin_summary.get(spin_type, 0) + 1

            dets = traj.detections
            results.append(SpinResult(
                rally_id=i + 1,
                frame_start=dets[0].frame_idx if dets else 0,
                frame_end=dets[-1].frame_idx if dets else 0,
                spin_type=spin_type,
                confidence=confidence,
                all_probs=all_probs,
                trajectory_points=traj_dict["positions"][:20],  # truncate for response size
                ball_speed_px_per_frame=traj_dict["mean_speed"],
                arc_direction=traj_dict["arc_direction"],
                lateral_drift=round(traj_dict["lateral_drift"], 3),
            ))

        return AnalysisResponse(
            job_id=job_id,
            video_filename=video.filename,
            processing_time_sec=round(time.time() - t0, 2),
            total_rallies=len(trajectories),
            spin_summary=spin_summary,
            results=results,
            model_ready=registry.is_ready,
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/analyze/frame")
async def analyze_frame(image: UploadFile = File(...)):
    """
    Upload a single frame image → detect ball position only.
    Useful for testing detection without full video.
    """
    cfg = registry.config or load_config()
    content = await video.read()
    img_array = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Cannot decode image")

    tracker = BallTracker(cfg)
    detections = tracker.detect(frame)

    return {
        "detections": [
            {
                "x_center": round(d.x_center, 4),
                "y_center": round(d.y_center, 4),
                "width": round(d.width, 4),
                "height": round(d.height, 4),
                "confidence": round(d.confidence, 4),
            }
            for d in detections
        ],
        "n_detections": len(detections),
    }


@app.websocket("/stream")
async def stream_inference(websocket: WebSocket):
    """
    WebSocket endpoint for real-time inference from webcam.

    Client sends base64-encoded JPEG frames.
    Server returns JSON with detection + classification results.

    Protocol:
      Client → base64(JPEG frame)
      Server → {"spin": "topspin", "confidence": 0.91, "ball": {"x": 0.5, "y": 0.3}}
    """
    await websocket.accept()
    cfg = registry.config or load_config()
    tracker = BallTracker(cfg)

    trajectory_buffer = []
    frame_count = 0

    logger.info(f"WebSocket connection opened")

    try:
        while True:
            data = await websocket.receive_text()

            try:
                img_bytes = base64.b64decode(data)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "Invalid frame data"})
                continue

            if frame is None:
                await websocket.send_json({"error": "Cannot decode frame"})
                continue

            # Detect
            detections = tracker.detect(frame)
            tracked = tracker.update(detections, frame)

            response = {
                "frame": frame_count,
                "detections": len(tracked),
                "ball": None,
                "spin": None,
                "confidence": None,
            }

            if tracked:
                d = tracked[0]
                response["ball"] = {
                    "x": round(d.x_center, 4),
                    "y": round(d.y_center, 4),
                    "w": round(d.width, 4),
                    "h": round(d.height, 4),
                    "conf": round(d.confidence, 4),
                }
                trajectory_buffer.append({"positions": [[d.x_center, d.y_center]]})

                # Classify when we have enough frames
                WINDOW = cfg["data"]["trajectory_window"]
                if len(trajectory_buffer) >= WINDOW:
                    positions = [p["positions"][0] for p in trajectory_buffer[-WINDOW:]]
                    traj_dict = {
                        "track_id": 0,
                        "positions": positions,
                        "spin_label": None,
                    }
                    spin, conf, probs = predict_spin(traj_dict)
                    response["spin"] = spin
                    response["confidence"] = conf
                    response["all_probs"] = probs
            else:
                # Gap in detection — reset buffer
                if len(trajectory_buffer) > 5:
                    trajectory_buffer = []

            frame_count += 1
            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


def start():
    import uvicorn
    cfg = load_config()
    api_cfg = cfg["api"]
    uvicorn.run("src.api.main:app", host=api_cfg["host"], port=api_cfg["port"], reload=False)


if __name__ == "__main__":
    start()
