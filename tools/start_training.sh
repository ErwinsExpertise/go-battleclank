#!/bin/bash
# 24/7 Training Starter Script with Auto-Recovery

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

LOG_DIR="nn_training_results"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "=================================================="
echo "🚀 Starting 24/7 Continuous Training"
echo "=================================================="
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop (will save checkpoint)"
echo "=================================================="
echo ""

# Run initial setup
echo "🔧 Running initial setup..."
./tools/quick_setup.sh

# Start continuous training with auto-restart on crash
while true; do
    echo ""
    echo "▶️  Starting training session..."
    python3 tools/continuous_training.py \
        --games 30 \
        --checkpoint-interval 10 \
        --min-improvement 0.001 \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Training completed normally"
        break
    elif [ $EXIT_CODE -eq 130 ]; then
        echo "⚠ Training interrupted by user (Ctrl+C)"
        break
    else
        echo "❌ Training crashed with exit code $EXIT_CODE"
        echo "⏳ Waiting 30 seconds before restart..."
        sleep 30
        echo "🔄 Restarting training..."
    fi
done

echo ""
echo "=================================================="
echo "🏁 Training stopped"
echo "Results saved in: $LOG_DIR"
echo "=================================================="
