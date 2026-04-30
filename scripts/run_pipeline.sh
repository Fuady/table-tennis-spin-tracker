#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# scripts/run_pipeline.sh
# Full end-to-end pipeline: data → features → train → evaluate → serve
# Usage: bash scripts/run_pipeline.sh [--synthetic]
# ─────────────────────────────────────────────────────────────────

set -e  # Exit on error
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()  { echo -e "${GREEN}✓ $1${NC}"; }
warn(){ echo -e "${YELLOW}⚠ $1${NC}"; }

SYNTHETIC=false
for arg in "$@"; do [[ "$arg" == "--synthetic" ]] && SYNTHETIC=true; done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  TT Spin Tracker — End-to-End Pipeline       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Step 1: Data ingestion ──────────────────────────────────────
log "Step 1/6: Data Ingestion"
if [ "$SYNTHETIC" = true ]; then
    warn "Using synthetic data (--synthetic flag)"
    python src/ingestion/download_ttnet.py --synthetic --output data/raw/ttnet
else
    log "Downloading TTNet sample dataset..."
    python src/ingestion/download_ttnet.py --output data/raw/ttnet
    log "Scraping YouTube table tennis videos..."
    python src/ingestion/youtube_scraper.py \
        --query "table tennis topspin training" \
        --max_videos 5 \
        --output data/raw/youtube
fi
ok "Data ingestion complete"

# ── Step 2: Frame extraction ────────────────────────────────────
log "Step 2/6: Frame Extraction"
python src/processing/extract_frames.py \
    --input data/raw/ \
    --output data/processed/ \
    --fps 30 \
    --resize 640 640 \
    --quality_filter 30.0
ok "Frames extracted → data/processed/frames/"

# ── Step 3: Ball tracking + trajectory extraction ───────────────
log "Step 3/6: Ball Tracking & Trajectory Extraction"
for video in data/raw/**/*.mp4; do
    if [ -f "$video" ]; then
        log "  Tracking: $video"
        python src/processing/tracker.py \
            --video "$video" \
            --output data/processed/trajectories/ \
            --min_len 8
    fi
done
ok "Trajectories saved → data/processed/trajectories/"

# ── Step 4: Feature engineering ────────────────────────────────
log "Step 4/6: Feature Engineering"
python src/processing/feature_engineering.py \
    --trajectories data/processed/trajectories/ \
    --output data/processed/features.parquet
ok "Features saved → data/processed/features.parquet"

# ── Step 5: Model training ──────────────────────────────────────
log "Step 5/6: Model Training (with MLflow tracking)"
python src/modeling/train_classifier.py \
    --config configs/config.yaml \
    --features data/processed/features.parquet \
    --experiment tt_spin_v1
ok "Model trained → models/spin_classifier_best.pt"

# ── Step 6: Evaluation ──────────────────────────────────────────
log "Step 6/6: Model Evaluation"
python src/modeling/evaluate.py \
    --model-path models/spin_classifier_best.pt \
    --features data/processed/features.parquet \
    --output reports/evaluation/
ok "Evaluation report → reports/evaluation/"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅ Pipeline Complete!                        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  → Start API:       uvicorn src.api.main:app --port 8000"
echo "  → Start Dashboard: streamlit run src/api/dashboard.py"
echo "  → View MLflow:     mlflow ui --port 5000"
echo ""
