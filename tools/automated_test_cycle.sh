#!/bin/bash
# Automated test cycle to reach 80% win rate
# Runs failure analysis, tests strategies, and implements best approach

set -e

cd "$(dirname "$0")/.."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="test_cycle_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          Automated Test Cycle for 80% Win Rate            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo
echo "Logs will be saved to: $LOG_DIR"
echo

# Step 1: Setup
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 1: Running Setup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
./tools/quick_setup.sh 2>&1 | tee "$LOG_DIR/01_setup.log"

# Step 2: Baseline performance check
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 2: Baseline Performance Check (30 games)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
python3 tools/run_benchmark.py 30 2>&1 | tee "$LOG_DIR/02_baseline_check.log"

BASELINE_WIN_RATE=$(grep -oP "Wins:.*?\(\K[0-9.]+(?=%)" "$LOG_DIR/02_baseline_check.log" | tail -1)
echo
echo -e "${YELLOW}Baseline Win Rate: ${BASELINE_WIN_RATE}%${NC}"
echo

# Step 3: Failure analysis
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 3: Failure Pattern Analysis (50 games)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
python3 tools/analyze_failures.py 50 2>&1 | tee "$LOG_DIR/03_failure_analysis.log"

# Step 4: Test alternative strategies
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 4: Testing Alternative Strategies${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo "This will test MCTS, Lookahead, and A-Star variations..."
echo "This may take 30-60 minutes..."
echo

python3 tools/test_strategies.py 2>&1 | tee "$LOG_DIR/04_strategy_testing.log"

# Step 5: Summary
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Test Cycle Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo
echo "Results Summary:"
echo "  Baseline: ${BASELINE_WIN_RATE}%"
echo "  Logs: $LOG_DIR/"
echo
echo "Next steps:"
echo "  1. Review logs in $LOG_DIR/"
echo "  2. Check strategy_test_results_*.json for best strategy"
echo "  3. Check failure_analysis_*.json for patterns"
echo "  4. Implement best performing strategy"
echo

# Check if we reached target
if (( $(echo "$BASELINE_WIN_RATE >= 80" | bc -l) )); then
    echo -e "${GREEN}🎉 TARGET ACHIEVED! Win rate >= 80%${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Target not yet reached. Review logs and iterate.${NC}"
    exit 1
fi
