#!/bin/bash
# Stop snake servers started by start_servers.sh

GO_PORT=${1:-8000}
RUST_PORT=${2:-8080}

echo "Stopping snake servers..."

# Stop Go snake
if [ -f /tmp/battlesnake_go_${GO_PORT}.pid ]; then
    GO_PID=$(cat /tmp/battlesnake_go_${GO_PORT}.pid)
    if kill -0 $GO_PID 2>/dev/null; then
        kill $GO_PID 2>/dev/null || true
        sleep 1
        if kill -0 $GO_PID 2>/dev/null; then
            kill -9 $GO_PID 2>/dev/null || true
        fi
        echo "  ✓ Stopped Go snake (PID: $GO_PID)"
    fi
    rm /tmp/battlesnake_go_${GO_PORT}.pid
fi

# Stop Rust baseline
if [ -f /tmp/battlesnake_rust_${RUST_PORT}.pid ]; then
    RUST_PID=$(cat /tmp/battlesnake_rust_${RUST_PORT}.pid)
    if kill -0 $RUST_PID 2>/dev/null; then
        kill $RUST_PID 2>/dev/null || true
        sleep 1
        if kill -0 $RUST_PID 2>/dev/null; then
            kill -9 $RUST_PID 2>/dev/null || true
        fi
        echo "  ✓ Stopped Rust baseline (PID: $RUST_PID)"
    fi
    rm /tmp/battlesnake_rust_${RUST_PORT}.pid
fi

echo "✓ Servers stopped"
