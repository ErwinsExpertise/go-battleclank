#!/bin/bash
# Quick setup script for running benchmarks and tests
# This ensures all dependencies are in place before testing

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Quick Setup for Benchmark Testing                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to repo root
cd "$(dirname "$0")/.."

echo "Step 1: Checking Go installation..."
if ! command -v go &> /dev/null; then
    echo "❌ Go not found!"
    exit 1
fi
echo -e "${GREEN}✓ Go found: $(go version)${NC}"

echo
echo "Step 2: Installing battlesnake CLI..."
if ! command -v "$(go env GOPATH)/bin/battlesnake" &> /dev/null; then
    echo "Installing battlesnake CLI..."
    go install github.com/BattlesnakeOfficial/rules/cli/battlesnake@latest
else
    echo -e "${GREEN}✓ battlesnake CLI already installed${NC}"
fi

echo
echo "Step 3: Building Rust baseline snake..."
if [ ! -f "baseline/target/release/baseline-snake" ]; then
    echo "Building baseline snake (this may take a few minutes)..."
    cd baseline && cargo build --release && cd ..
else
    echo -e "${GREEN}✓ Baseline snake already built${NC}"
fi

echo
echo "Step 4: Building Go snake..."
go build -o go-battleclank
echo -e "${GREEN}✓ Go snake built${NC}"

echo
echo "Step 5: Making scripts executable..."
chmod +x tools/*.py tools/*.sh 2>/dev/null || true
echo -e "${GREEN}✓ Scripts ready${NC}"

echo
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo
echo "Available commands:"
echo "  ${YELLOW}python3 tools/run_benchmark.py 100${NC}        - Run 100 games benchmark"
echo "  ${YELLOW}python3 tools/analyze_failures.py 50${NC}      - Analyze failure patterns"
echo "  ${YELLOW}python3 tools/test_strategies.py${NC}          - Test MCTS/Lookahead/A-Star"
echo "  ${YELLOW}python3 tools/optimize_weights.py${NC}         - Optimize weights"
echo "  ${YELLOW}./tools/automated_test_cycle.sh${NC}           - Run full test cycle"
echo
