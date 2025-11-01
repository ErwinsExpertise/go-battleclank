#!/bin/bash

# Live benchmark script to test Go snake vs actual Rust baseline snake
# Uses the battlesnake CLI to run real games

set -e

# Configuration
NUM_GAMES=${1:-100}
BOARD_WIDTH=${2:-11}
BOARD_HEIGHT=${3:-11}
MAX_TURNS=${4:-500}
OUTPUT_DIR="./benchmark_results_live"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$OUTPUT_DIR/results_${TIMESTAMP}.json"
SUMMARY_FILE="$OUTPUT_DIR/summary_${TIMESTAMP}.txt"
DETAILS_FILE="$OUTPUT_DIR/details_${TIMESTAMP}.jsonl"

# Ports
GO_PORT=8000
RUST_PORT=8080

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================"
echo "  Live Battlesnake Benchmark"
echo "  Go Snake vs Rust Baseline"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Games to run: $NUM_GAMES"
echo "  Board size: ${BOARD_WIDTH}x${BOARD_HEIGHT}"
echo "  Max turns: $MAX_TURNS"
echo "  Go snake: http://localhost:$GO_PORT"
echo "  Rust baseline: http://localhost:$RUST_PORT"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if battlesnake CLI is installed
if ! command -v battlesnake &> /dev/null; then
    echo -e "${RED}Error: battlesnake CLI not found${NC}"
    echo "Installing battlesnake CLI..."
    go install github.com/BattlesnakeOfficial/rules/cli/battlesnake@latest
    export PATH=$PATH:$(go env GOPATH)/bin
    if ! command -v battlesnake &> /dev/null; then
        echo -e "${RED}Failed to install battlesnake CLI${NC}"
        exit 1
    fi
fi

# Build both snakes
echo "Building snakes..."
echo -n "  Building Go snake... "
go build -o go-battleclank . || { echo -e "${RED}FAILED${NC}"; exit 1; }
echo -e "${GREEN}OK${NC}"

echo -n "  Building Rust baseline... "
cd baseline && cargo build --release --quiet || { echo -e "${RED}FAILED${NC}"; exit 1; }
cd ..
echo -e "${GREEN}OK${NC}"

# Start both snakes in background
echo ""
echo "Starting snake servers..."

# Start Go snake
echo -n "  Starting Go snake on port $GO_PORT... "
PORT=$GO_PORT ./go-battleclank > /tmp/go-snake.log 2>&1 &
GO_PID=$!
sleep 2
if ! kill -0 $GO_PID 2>/dev/null; then
    echo -e "${RED}FAILED${NC}"
    echo "Go snake failed to start. Check /tmp/go-snake.log"
    exit 1
fi
echo -e "${GREEN}OK${NC} (PID: $GO_PID)"

# Start Rust baseline
echo -n "  Starting Rust baseline on port $RUST_PORT... "
BIND_PORT=$RUST_PORT ./baseline/target/release/baseline-snake > /tmp/rust-baseline.log 2>&1 &
RUST_PID=$!
sleep 2
if ! kill -0 $RUST_PID 2>/dev/null; then
    echo -e "${RED}FAILED${NC}"
    echo "Rust baseline failed to start. Check /tmp/rust-baseline.log"
    kill $GO_PID 2>/dev/null || true
    exit 1
fi
echo -e "${GREEN}OK${NC} (PID: $RUST_PID)"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping snake servers..."
    kill $GO_PID 2>/dev/null || true
    kill $RUST_PID 2>/dev/null || true
    sleep 1
}

# Register cleanup on exit
trap cleanup EXIT

# Wait for servers to be ready
echo ""
echo "Waiting for servers to be ready..."
sleep 3

# Verify both snakes are responding
echo -n "  Checking Go snake... "
if curl -s http://localhost:$GO_PORT/ > /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}NOT RESPONDING${NC}"
    exit 1
fi

echo -n "  Checking Rust baseline... "
if curl -s http://localhost:$RUST_PORT/ > /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}NOT RESPONDING${NC}"
    exit 1
fi

# Initialize counters
wins=0
losses=0
draws=0
go_deaths_starve=0
go_deaths_collision=0
go_deaths_headtohead=0
go_deaths_timeout=0
go_deaths_outofbounds=0
go_deaths_eliminated=0
total_turns=0
total_go_length=0
total_go_food=0

# Create details file header
echo "[]" > "$DETAILS_FILE.tmp"

echo ""
echo "================================================"
echo "  Running Games"
echo "================================================"
echo ""

# Run games
for i in $(seq 1 $NUM_GAMES); do
    progress=$(awk "BEGIN {printf \"%.1f\", $i/$NUM_GAMES*100}")
    echo -ne "\rGame $i/$NUM_GAMES ($progress%)... "
    
    # Run game and save output
    game_output=$(mktemp)
    
    if battlesnake play \
        -W $BOARD_WIDTH \
        -H $BOARD_HEIGHT \
        -t $MAX_TURNS \
        --name "go-battleclank" --url "http://localhost:$GO_PORT" \
        --name "rust-baseline" --url "http://localhost:$RUST_PORT" \
        --sequential \
        --output json \
        --seed "$RANDOM" \
        2>/dev/null > "$game_output"; then
        
        # Parse game result
        winner=$(jq -r '.Winner // "draw"' "$game_output" 2>/dev/null || echo "error")
        
        if [ "$winner" = "go-battleclank" ]; then
            wins=$((wins + 1))
            echo -e "${GREEN}WIN${NC}"
        elif [ "$winner" = "rust-baseline" ]; then
            losses=$((losses + 1))
            
            # Determine cause of death from game JSON
            death_reason=$(jq -r '.Snakes[] | select(.Name == "go-battleclank") | .Death.Cause // "unknown"' "$game_output" 2>/dev/null || echo "unknown")
            
            case "$death_reason" in
                "starvation")
                    go_deaths_starve=$((go_deaths_starve + 1))
                    echo -e "${RED}LOSS${NC} (starvation)"
                    ;;
                "snake-collision"|"self-collision")
                    go_deaths_collision=$((go_deaths_collision + 1))
                    echo -e "${RED}LOSS${NC} (collision)"
                    ;;
                "head-collision")
                    go_deaths_headtohead=$((go_deaths_headtohead + 1))
                    echo -e "${RED}LOSS${NC} (head-to-head)"
                    ;;
                "wall-collision"|"out-of-bounds")
                    go_deaths_outofbounds=$((go_deaths_outofbounds + 1))
                    echo -e "${RED}LOSS${NC} (out of bounds)"
                    ;;
                *)
                    go_deaths_eliminated=$((go_deaths_eliminated + 1))
                    echo -e "${RED}LOSS${NC} ($death_reason)"
                    ;;
            esac
        else
            draws=$((draws + 1))
            echo -e "${YELLOW}DRAW${NC}"
        fi
        
        # Extract metrics
        turns=$(jq -r '.Turn // 0' "$game_output" 2>/dev/null || echo "0")
        go_snake=$(jq '.Snakes[] | select(.Name == "go-battleclank")' "$game_output" 2>/dev/null || echo '{}')
        go_length=$(echo "$go_snake" | jq -r '.Body | length' 2>/dev/null || echo "0")
        
        total_turns=$((total_turns + turns))
        total_go_length=$((total_go_length + go_length))
        
        # Append game details to file
        jq -c ". + {game_number: $i}" "$game_output" >> "$DETAILS_FILE.tmp" 2>/dev/null || true
        
    else
        echo -e "${YELLOW}ERROR${NC}"
        draws=$((draws + 1))
    fi
    
    rm -f "$game_output"
done

echo ""
echo "================================================"
echo "  Benchmark Results Summary"
echo "================================================"
echo ""
echo "Games Played: $NUM_GAMES"

win_rate=$(awk "BEGIN {printf \"%.1f\", $wins/$NUM_GAMES*100}")
loss_rate=$(awk "BEGIN {printf \"%.1f\", $losses/$NUM_GAMES*100}")

echo -e "Wins:   ${GREEN}$wins${NC} ($win_rate%)"
echo -e "Losses: ${RED}$losses${NC} ($loss_rate%)"
echo "Draws:  $draws"
echo ""
echo "Cause of Death (Go Snake):"
echo "  Starvation:     $go_deaths_starve"
echo "  Collision:      $go_deaths_collision"
echo "  Head-to-head:   $go_deaths_headtohead"
echo "  Out of bounds:  $go_deaths_outofbounds"
echo "  Eliminated:     $go_deaths_eliminated"
echo ""
echo "Performance Metrics:"
if [ $NUM_GAMES -gt 0 ]; then
    avg_turns=$(awk "BEGIN {printf \"%.1f\", $total_turns/$NUM_GAMES}")
    avg_length=$(awk "BEGIN {printf \"%.1f\", $total_go_length/$NUM_GAMES}")
    echo "  Average turns survived: $avg_turns"
    echo "  Average final length:   $avg_length"
fi
echo ""

# Create JSON results
cat > "$RESULTS_FILE" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "config": {
    "num_games": $NUM_GAMES,
    "board_width": $BOARD_WIDTH,
    "board_height": $BOARD_HEIGHT,
    "max_turns": $MAX_TURNS
  },
  "results": {
    "wins": $wins,
    "losses": $losses,
    "draws": $draws,
    "win_rate": $win_rate
  },
  "death_causes": {
    "starvation": $go_deaths_starve,
    "collision": $go_deaths_collision,
    "head_to_head": $go_deaths_headtohead,
    "out_of_bounds": $go_deaths_outofbounds,
    "eliminated": $go_deaths_eliminated
  },
  "metrics": {
    "avg_turns": $avg_turns,
    "avg_final_length": $avg_length
  }
}
EOF

# Write summary to file
{
    echo "Live Battlesnake Benchmark Results"
    echo "==================================="
    echo "Timestamp: $TIMESTAMP"
    echo "Games: $NUM_GAMES"
    echo "Board: ${BOARD_WIDTH}x${BOARD_HEIGHT}"
    echo "Max Turns: $MAX_TURNS"
    echo ""
    echo "Win Rate: $win_rate%"
    echo "Wins: $wins"
    echo "Losses: $losses"
    echo "Draws: $draws"
    echo ""
    echo "Go Snake Deaths:"
    echo "  Starvation: $go_deaths_starve"
    echo "  Collision: $go_deaths_collision"
    echo "  Head-to-head: $go_deaths_headtohead"
    echo "  Out of bounds: $go_deaths_outofbounds"
    echo "  Eliminated: $go_deaths_eliminated"
    echo ""
    echo "Performance:"
    echo "  Avg Turns: $avg_turns"
    echo "  Avg Length: $avg_length"
} > "$SUMMARY_FILE"

# Process details file
if [ -f "$DETAILS_FILE.tmp" ]; then
    mv "$DETAILS_FILE.tmp" "$DETAILS_FILE"
fi

echo "Results saved to:"
echo "  Summary: $SUMMARY_FILE"
echo "  JSON:    $RESULTS_FILE"
echo "  Details: $DETAILS_FILE"
echo ""

# Assessment
if [ $(echo "$win_rate >= 80" | bc -l 2>/dev/null || echo "0") -eq 1 ]; then
    echo -e "${GREEN}✓ SUCCESS: Win rate >= 80% target!${NC}"
    exit 0
elif [ $(echo "$win_rate >= 60" | bc -l 2>/dev/null || echo "0") -eq 1 ]; then
    echo -e "${YELLOW}⚠ PARTIAL: Win rate >= 60% (competitive parity)${NC}"
    exit 0
else
    echo -e "${RED}✗ BELOW TARGET: Win rate < 60%${NC}"
    exit 1
fi
