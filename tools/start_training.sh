#!/bin/bash
# 24/7 Training Starter Script with Auto-Recovery
#
# Usage: ./start_training.sh [parallel_configs]
#   parallel_configs: Number of parallel configurations to test (default: 1)
#                     For 8x A100 GPUs, use 8 for maximum utilization
#
# Examples:
#   ./start_training.sh      # Single configuration (sequential)
#   ./start_training.sh 4    # 4 parallel configurations
#   ./start_training.sh 8    # 8 parallel configurations (for 8 GPUs)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

LOG_DIR="nn_training_results"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

# Parse arguments for parallel configs
PARALLEL_CONFIGS=${1:-1}

echo "=================================================="
echo "🚀 Starting 24/7 Continuous Training"
echo "=================================================="
echo "Parallel configurations: $PARALLEL_CONFIGS"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop (will save checkpoint)"
echo "=================================================="
echo ""

# Run initial setup
echo "🔧 Running initial setup..."
./tools/quick_setup.sh

# Start snake servers (they'll stay running for continuous benchmarking)
echo ""
echo "🐍 Starting snake servers for $PARALLEL_CONFIGS parallel configurations..."
for i in $(seq 0 $((PARALLEL_CONFIGS - 1))); do
    GO_PORT=$((8000 + i * 100))
    RUST_PORT=$((8080 + i * 100))
    echo "  Starting server pair $((i + 1))/$PARALLEL_CONFIGS (ports: $GO_PORT, $RUST_PORT)..."
    ./tools/start_servers.sh $GO_PORT $RUST_PORT
done
echo ""

# Setup cleanup on exit
cleanup_servers() {
    echo ""
    echo "🛑 Stopping all server pairs..."
    for i in $(seq 0 $((PARALLEL_CONFIGS - 1))); do
        GO_PORT=$((8000 + i * 100))
        RUST_PORT=$((8080 + i * 100))
        ./tools/stop_servers.sh $GO_PORT $RUST_PORT
    done
}
trap cleanup_servers EXIT

# Start continuous training with auto-restart on crash
while true; do
    echo ""
    echo "▶️  Starting training session with $PARALLEL_CONFIGS parallel configurations..."
    python3 tools/continuous_training.py \
        --games 100 \
        --checkpoint-interval 10 \
        --min-improvement 0.1 \
        --parallel-configs $PARALLEL_CONFIGS \
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
