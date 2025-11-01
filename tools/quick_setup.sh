#!/bin/bash
# Quick setup script for running benchmarks and tests
# This ensures all dependencies are in place before testing
# Tested on Ubuntu 22.04 and 24.04

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Quick Setup for Benchmark Testing                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Navigate to repo root
cd "$(dirname "$0")/.."

echo "Step 1: Checking system dependencies..."

# Check Python3 and pip3
NEED_INSTALL=false
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found!${NC}"
    NEED_INSTALL=true
else
    echo -e "${GREEN}✓ Python3 found: $(python3 --version)${NC}"
fi

if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 not found!${NC}"
    NEED_INSTALL=true
else
    echo -e "${GREEN}✓ pip3 found${NC}"
fi

# Install Python3 and pip3 if needed (single apt-get update)
if [ "$NEED_INSTALL" = true ]; then
    echo "Installing Python3 and pip3..."
    sudo apt-get update && sudo apt-get install -y python3 python3-pip
fi

# Check Go
if ! command -v go &> /dev/null; then
    echo -e "${RED}❌ Go not found!${NC}"
    echo "Please install Go from https://golang.org/dl/"
    exit 1
fi
echo -e "${GREEN}✓ Go found: $(go version)${NC}"

# Check Rust/Cargo
if ! command -v cargo &> /dev/null; then
    echo -e "${YELLOW}⚠ Rust/Cargo not found!${NC}"
    echo "To install Rust, please run:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    echo "  source \$HOME/.cargo/env"
    echo "Or install via your package manager (e.g., 'sudo apt install cargo' on Ubuntu 24.04+)"
    exit 1
else
    echo -e "${GREEN}✓ Rust/Cargo found: $(cargo --version)${NC}"
fi

echo
echo "Step 2: Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt (user-local)..."
    pip3 install --user -r requirements.txt --quiet
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ requirements.txt not found, skipping...${NC}"
fi

echo
echo "Step 3: Installing battlesnake CLI..."
GOPATH=$(go env GOPATH)
if [ ! -f "${GOPATH}/bin/battlesnake" ]; then
    echo "Installing battlesnake CLI..."
    go install github.com/BattlesnakeOfficial/rules/cli/battlesnake@latest
    echo -e "${GREEN}✓ battlesnake CLI installed${NC}"
else
    echo -e "${GREEN}✓ battlesnake CLI already installed${NC}"
fi

echo
echo "Step 4: Building Rust baseline snake..."
if [ ! -f "baseline/target/release/baseline-snake" ]; then
    echo "Building baseline snake (this may take a few minutes)..."
    cd baseline
    cargo build --release
    cd ..
    if [ ! -f "baseline/target/release/baseline-snake" ]; then
        echo -e "${RED}❌ Failed to build Rust baseline snake${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Baseline snake built successfully${NC}"
else
    echo -e "${GREEN}✓ Baseline snake already built${NC}"
fi

echo
echo "Step 5: Building Go snake..."
go build -o go-battleclank
if [ ! -f "go-battleclank" ]; then
    echo -e "${RED}❌ Failed to build Go snake${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Go snake built successfully${NC}"

echo
echo "Step 6: Making scripts executable..."
chmod +x tools/*.py tools/*.sh 2>/dev/null || true
echo -e "${GREEN}✓ Scripts ready${NC}"

echo
echo "Step 7: Verifying setup..."
echo "Checking binaries..."
if [ -f "go-battleclank" ] && [ -f "baseline/target/release/baseline-snake" ]; then
    echo -e "${GREEN}✓ All binaries present${NC}"
else
    echo -e "${RED}❌ Some binaries missing${NC}"
    [ ! -f "go-battleclank" ] && echo "  - go-battleclank not found"
    [ ! -f "baseline/target/release/baseline-snake" ] && echo "  - baseline-snake not found"
    exit 1
fi

echo
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo
echo "Available commands:"
echo "  ${YELLOW}python3 tools/run_benchmark.py 100${NC}        - Run 100 games benchmark"
echo "  ${YELLOW}./tools/start_training.sh${NC}                 - Start 24/7 training"
echo "  ${YELLOW}python3 tools/analyze_failures.py 50${NC}      - Analyze failure patterns"
echo "  ${YELLOW}python3 tools/test_strategies.py${NC}          - Test MCTS/Lookahead/A-Star"
echo "  ${YELLOW}python3 tools/optimize_weights.py${NC}         - Optimize weights"
echo "  ${YELLOW}./tools/automated_test_cycle.sh${NC}           - Run full test cycle"
echo
