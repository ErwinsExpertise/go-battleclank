#!/bin/bash
# Start snake servers for benchmarking
# This script starts servers and keeps them running for continuous benchmarking

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

GO_PORT=${1:-8000}
RUST_PORT=${2:-8080}

echo "Starting snake servers..."

# Start Go snake
echo "  Starting Go snake on port $GO_PORT..."
PORT=$GO_PORT ./battlesnake > /dev/null 2>&1 &
GO_PID=$!
echo $GO_PID > /tmp/battlesnake_go_${GO_PORT}.pid
sleep 2

# Check if Go snake started successfully
if ! kill -0 $GO_PID 2>/dev/null; then
    echo "  ✗ Failed to start Go snake"
    exit 1
fi
echo "  ✓ Go snake started (PID: $GO_PID)"

# Start Rust baseline
echo "  Starting Rust baseline on port $RUST_PORT..."
BIND_PORT=$RUST_PORT ./baseline/target/release/baseline-snake > /dev/null 2>&1 &
RUST_PID=$!
echo $RUST_PID > /tmp/battlesnake_rust_${RUST_PORT}.pid
sleep 2

# Check if Rust snake started successfully
if ! kill -0 $RUST_PID 2>/dev/null; then
    echo "  ✗ Failed to start Rust snake"
    kill $GO_PID 2>/dev/null || true
    rm /tmp/battlesnake_go_${GO_PORT}.pid 2>/dev/null || true
    exit 1
fi
echo "  ✓ Rust baseline started (PID: $RUST_PID)"

echo ""
echo "✓ Both servers are running"
echo "  Go snake:      http://localhost:$GO_PORT (PID: $GO_PID)"
echo "  Rust baseline: http://localhost:$RUST_PORT (PID: $RUST_PID)"
