#!/bin/bash
# Download Spider dataset (if not already present)
set -euo pipefail

DATA_DIR="data/spider_data"

if [ -d "$DATA_DIR/spider_data" ]; then
    echo "Spider dataset already exists at $DATA_DIR/spider_data"
    echo "Skipping download."
    exit 0
fi

mkdir -p "$DATA_DIR"

echo "Downloading Spider dataset..."
# Option 1: From the official Spider GitHub
wget -q "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ" \
    -O "$DATA_DIR/spider.zip" 2>/dev/null || \
    echo "Direct download failed. Please download Spider manually from https://yale-lily.github.io/spider"

if [ -f "$DATA_DIR/spider.zip" ]; then
    echo "Extracting..."
    unzip -q "$DATA_DIR/spider.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/spider.zip"
    echo "Spider dataset extracted to $DATA_DIR/spider_data"
else
    echo "Please download Spider manually:"
    echo "  1. Visit https://yale-lily.github.io/spider"
    echo "  2. Download the dataset"
    echo "  3. Extract to $DATA_DIR/spider_data"
fi

# Verify
if [ -f "$DATA_DIR/spider_data/train_spider.json" ]; then
    echo "Verification: train_spider.json found"
    python -c "import json; d=json.load(open('$DATA_DIR/spider_data/train_spider.json')); print(f'  {len(d)} training examples')"
else
    echo "WARNING: train_spider.json not found. Dataset may not be properly extracted."
fi
