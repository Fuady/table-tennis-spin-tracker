#!/bin/bash
# scripts/download_data.sh
# Download all required data sources

set -e
CYAN='\033[0;36m'; GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()  { echo -e "${GREEN}✓ $1${NC}"; }

echo "─────────────────────────────────────────────"
echo " TT Spin Tracker — Data Download"
echo "─────────────────────────────────────────────"

mkdir -p data/raw/{ttnet,youtube,roboflow}

# 1. TTNet sample dataset
log "Downloading TTNet sample dataset..."
python src/ingestion/download_ttnet.py \
    --source ttnet \
    --output data/raw/ttnet

# 2. Synthetic samples (always create for testing)
log "Creating synthetic test samples..."
python src/ingestion/download_ttnet.py \
    --synthetic \
    --output data/raw/synthetic

# 3. YouTube
log "Downloading YouTube videos..."
queries=(
    "table tennis topspin training"
    "table tennis backspin serve slow motion"
    "table tennis sidespin loop"
)
for q in "${queries[@]}"; do
    python src/ingestion/youtube_scraper.py \
        --query "$q" \
        --max_videos 5 \
        --output "data/raw/youtube/${q// /_}"
done

ok "All data downloaded!"
echo ""
echo "Data locations:"
echo "  data/raw/ttnet/     — TTNet annotated dataset"
echo "  data/raw/synthetic/ — Synthetic test videos"
echo "  data/raw/youtube/   — YouTube scraped videos"
echo ""
echo "Next: bash scripts/run_pipeline.sh"
