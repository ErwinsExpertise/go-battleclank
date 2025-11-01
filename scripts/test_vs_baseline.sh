#!/bin/bash

# Benchmarking script to test Go snake vs Rust baseline snake
# Requires: battlesnake CLI tool (https://github.com/BattlesnakeOfficial/rules)

set -e

# Configuration
NUM_GAMES=${1:-100}
BOARD_WIDTH=${2:-11}
BOARD_HEIGHT=${3:-11}
OUTPUT_DIR="./benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$OUTPUT_DIR/results_${TIMESTAMP}.json"
SUMMARY_FILE="$OUTPUT_DIR/summary_${TIMESTAMP}.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "  Battlesnake Benchmark: Go vs Rust Baseline"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Games to run: $NUM_GAMES"
echo "  Board size: ${BOARD_WIDTH}x${BOARD_HEIGHT}"
echo "  Results directory: $OUTPUT_DIR"
echo ""

# Check if battlesnake CLI is installed
if ! command -v battlesnake &> /dev/null; then
    echo -e "${RED}Error: battlesnake CLI not found${NC}"
    echo "Please install from: https://github.com/BattlesnakeOfficial/rules"
    echo ""
    echo "Quick install:"
    echo "  go install github.com/BattlesnakeOfficial/rules/cli/battlesnake@latest"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize counters
wins=0
losses=0
draws=0
go_deaths_starve=0
go_deaths_collision=0
go_deaths_headtohead=0
go_deaths_timeout=0
total_turns=0
total_go_length=0
total_go_food=0

echo "Starting benchmark..."
echo ""

# Run games
for i in $(seq 1 $NUM_GAMES); do
    echo -n "Game $i/$NUM_GAMES: "
    
    # Run game in CLI mode (requires both snakes to be running on different ports)
    # This is a placeholder - actual implementation depends on how baseline snake is set up
    # For now, we'll create a simulated result
    
    # TODO: Implement actual game running with battlesnake CLI
    # Example command (when both snakes are running):
    # battlesnake play -W $BOARD_WIDTH -H $BOARD_HEIGHT \
    #   --name "go-snake" --url http://localhost:8000 \
    #   --name "rust-baseline" --url http://localhost:8080 \
    #   --sequential --output json > "$OUTPUT_DIR/game_${i}.json"
    
    echo -e "${YELLOW}[SIMULATED]${NC} Result pending actual implementation"
done

# Calculate statistics
echo ""
echo "================================================"
echo "  Benchmark Results Summary"
echo "================================================"
echo ""
echo "Games Played: $NUM_GAMES"
echo -e "Wins: ${GREEN}$wins${NC} ($(awk "BEGIN {printf \"%.1f\", $wins/$NUM_GAMES*100}")%)"
echo -e "Losses: ${RED}$losses${NC} ($(awk "BEGIN {printf \"%.1f\", $losses/$NUM_GAMES*100}")%)"
echo "Draws: $draws"
echo ""
echo "Cause of Death (Go Snake):"
echo "  Starvation: $go_deaths_starve"
echo "  Collision: $go_deaths_collision"
echo "  Head-to-head: $go_deaths_headtohead"
echo "  Timeout: $go_deaths_timeout"
echo ""
echo "Performance Metrics:"
if [ $NUM_GAMES -gt 0 ]; then
    avg_turns=$(awk "BEGIN {printf \"%.1f\", $total_turns/$NUM_GAMES}")
    avg_length=$(awk "BEGIN {printf \"%.1f\", $total_go_length/$NUM_GAMES}")
    avg_food=$(awk "BEGIN {printf \"%.1f\", $total_go_food/$NUM_GAMES}")
    echo "  Average turns survived: $avg_turns"
    echo "  Average final length: $avg_length"
    echo "  Average food collected: $avg_food"
fi
echo ""
echo "Results saved to:"
echo "  $SUMMARY_FILE"
echo ""

# Write summary to file
{
    echo "Battlesnake Benchmark Results"
    echo "=============================="
    echo "Timestamp: $TIMESTAMP"
    echo "Games: $NUM_GAMES"
    echo "Board: ${BOARD_WIDTH}x${BOARD_HEIGHT}"
    echo ""
    echo "Win Rate: $(awk "BEGIN {printf \"%.1f\", $wins/$NUM_GAMES*100}")%"
    echo "Wins: $wins"
    echo "Losses: $losses"
    echo "Draws: $draws"
    echo ""
    echo "Go Snake Deaths:"
    echo "  Starvation: $go_deaths_starve"
    echo "  Collision: $go_deaths_collision"
    echo "  Head-to-head: $go_deaths_headtohead"
    echo "  Timeout: $go_deaths_timeout"
} > "$SUMMARY_FILE"

echo "================================================"
echo ""

# Exit with success if win rate >= 80%
win_rate=$(awk "BEGIN {printf \"%.0f\", $wins/$NUM_GAMES*100}")
if [ "$win_rate" -ge 80 ]; then
    echo -e "${GREEN}✓ SUCCESS: Win rate >= 80% target!${NC}"
    exit 0
elif [ "$win_rate" -ge 60 ]; then
    echo -e "${YELLOW}⚠ PARTIAL: Win rate >= 60% (competitive parity)${NC}"
    exit 0
else
    echo -e "${RED}✗ BELOW TARGET: Win rate < 60%${NC}"
    exit 1
fi
