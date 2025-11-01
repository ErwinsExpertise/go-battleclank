# Battlesnake Training Guide

## Overview

This guide explains how to train and tune the Battlesnake configuration for optimal performance.

## Configuration System

All Battlesnake behavior is controlled by `config.yaml`. The configuration is loaded when the server starts and affects:

- Search algorithm selection (greedy, lookahead, MCTS, hybrid)
- Weight values for heuristics (space, food, collision avoidance)
- Trap detection thresholds
- Pursuit behavior parameters
- Late game and critical situation thresholds

### Configuration File Structure

```yaml
# Search algorithm configuration
search:
  algorithm: hybrid          # Options: greedy, lookahead, mcts, hybrid
  max_depth: 121            # Maximum search depth
  use_astar: true           # Use A* for pathfinding
  max_astar_nodes: 400      # A* node limit

# Heuristic weights
weights:
  space: 5.0                # Weight for available space
  head_collision: 500.0     # Penalty for head-to-head collision risk
  center_control: 2.0       # Bonus for center position
  wall_penalty: 5.0         # Penalty for being near walls
  cutoff: 200.0            # Penalty for dead ends
  food: 1.0                # Base food seeking weight

# Trap detection penalties
traps:
  moderate: 250.0          # Moderate trap penalty
  severe: 450.0            # Severe trap penalty
  critical: 600.0          # Critical trap penalty
  food_trap: 800.0         # Food death trap penalty
  food_trap_threshold: 0.7 # Threshold for food trap detection (0.0-1.0)

# Pursuit behavior (when longer than enemy)
pursuit:
  distance_2: 100.0        # Pursuit bonus at distance 2
  distance_3: 50.0         # Pursuit bonus at distance 3
  distance_4: 25.0         # Pursuit bonus at distance 4
  distance_5: 10.0         # Pursuit bonus at distance 5

# Trapping behavior (when trying to trap enemies)
trapping:
  weight: 400.0            # Base trapping bonus
  space_cutoff_threshold: 0.2  # Minimum space reduction to consider (0.0-1.0)
  trapped_ratio: 0.6       # Space/length ratio threshold for trapped enemy

# Food urgency multipliers
food_urgency:
  critical: 1.8            # Multiplier when health < 30
  low: 1.4                 # Multiplier when health < 50
  normal: 1.0              # Multiplier when health >= 50

# Late game behavior
late_game:
  turn_threshold: 150      # Turn count for late game
  caution_multiplier: 1.1  # Risk multiplier in late game

# Hybrid algorithm configuration
hybrid:
  use_lookahead_on_critical: true  # Use lookahead in critical situations
  lookahead_depth: 3               # Depth for lookahead search
  use_mcts_in_endgame: true        # Use MCTS in 1v1 endgame
  mcts_iterations: 100             # MCTS iteration limit
  mcts_timeout_ms: 200             # MCTS time limit (milliseconds)
  critical_health: 30              # Health threshold for critical
  critical_space_ratio: 3.0        # Space/length ratio for critical
  critical_nearby_enemies: 2       # Enemy count for critical
```

## Training Workflow

### 1. Baseline Testing

Before making changes, establish a baseline win rate:

```bash
# Build both snakes
go build -o go-battleclank
cd baseline && cargo build --release && cd ..

# Run benchmark with current config
python3 tools/run_benchmark.py 50
```

Target performance: ~47% win rate against rust baseline opponent (competitive parity)

### 2. Modifying Configuration

Edit `config.yaml` to adjust parameters. Guidelines:

**Weight Tuning:**
- Increase `space` (5.0 → 7.0) for more defensive play
- Increase `food` (1.0 → 1.5) for more aggressive food seeking
- Adjust `head_collision` (500.0) to control risk tolerance

**Trap Tuning:**
- Increase trap penalties to avoid risky positions
- Adjust `food_trap_threshold` (0.7) to be more/less cautious about food traps

**Pursuit Tuning:**
- Increase pursuit bonuses to be more aggressive when longer
- Decrease to be more cautious

**Example: More Aggressive Configuration**
```yaml
weights:
  space: 4.0              # Less defensive
  food: 1.2               # More food seeking
  
pursuit:
  distance_2: 120.0       # More aggressive pursuit
  distance_3: 60.0
  
food_urgency:
  low: 1.6                # More urgent food seeking
```

### 3. Testing Changes

After modifying config:

```bash
# Rebuild (optional - only if code changed)
go build -o battlesnake

# Restart server with new config
./battlesnake
```

The server will log the loaded configuration:
```
✓ Configuration loaded successfully:
  - Algorithm: hybrid
  - Space weight: 4.0
  - Head collision weight: 500.0
  - Food trap penalty: 800.0
  - Pursuit distance 2: 120.0
  - Trapping weight: 400.0
```

### 4. Benchmark Testing

Run benchmarks to evaluate changes:

```bash
# Build the Go snake
go build -o go-battleclank

# Build the Rust baseline (first time only)
cd baseline && cargo build --release && cd ..

# Quick test (10 games vs rust baseline)
python3 tools/run_benchmark.py 10

# Full test (50 games)
python3 tools/run_benchmark.py 50

# Extended test (100 games)
python3 tools/run_benchmark.py 100
```

**Note:** The `tools/run_benchmark.py` script runs actual games against the rust baseline snake using the Battlesnake CLI. This is the correct benchmark to use for evaluating performance.

### 5. Iterative Improvement

1. Run baseline benchmark (100 games)
2. Identify weakness (e.g., too many starvation deaths)
3. Adjust related config parameters
4. Restart server
5. Run benchmark again
6. Compare results
7. Keep changes if improved, revert if worse
8. Commit good configurations to git

## Training Best Practices

### Small Changes
- Adjust 1-3 parameters at a time
- Make incremental changes (10-20% adjustments)
- Test thoroughly before making larger changes

### Statistical Significance
- Run at least 50-100 games per test
- Look for consistent improvements (5%+ win rate increase)
- Consider variability in results

### Parameter Relationships
Some parameters interact:
- Higher `space` weight may reduce food seeking effectiveness
- Lower trap penalties may increase aggression but risk traps
- Higher pursuit bonuses work best with high health

### Common Issues

**Too Defensive (passive play):**
- Decrease `space` weight
- Increase `food` weight
- Increase pursuit bonuses
- Decrease trap penalties slightly

**Too Aggressive (risky play):**
- Increase `space` weight
- Increase trap penalties
- Decrease pursuit bonuses
- Increase `caution_multiplier` in late game

**Starvation Deaths:**
- Increase `food_urgency` multipliers
- Increase `food` base weight
- Decrease `food_trap` penalty (take more risks for food)

**Head-to-Head Losses:**
- Increase `head_collision` penalty
- Increase `pursuit` bonuses (be more aggressive when longer)
- Adjust `critical_nearby_enemies` threshold

## Automated Training

For automated parameter exploration, use the continuous training system:

```bash
# Start 24/7 training (requires battlesnake CLI installed)
python3 tools/continuous_training.py

# Training will:
# 1. Test current configuration
# 2. Generate parameter variations
# 3. Test variations
# 4. Keep improvements
# 5. Commit winning configurations
# 6. Repeat indefinitely
```

## Verifying Config Loading

To verify config is being read correctly:

```bash
# Test config reload without rebuild
./test_config_reload.sh
```

This script:
1. Starts server and logs config
2. Modifies config.yaml
3. Restarts server
4. Verifies new values loaded
5. Restores original config

## Environment Variables

- `BATTLESNAKE_CONFIG`: Override config file path (default: `config.yaml`)
- `PORT`: Server port (default: `8000`)
- `USE_LEGACY`: Use legacy logic (set to `true` to enable)

## Success Criteria

A successful training session should achieve:
- ✅ Benchmark win rate: ~47% against rust baseline (competitive parity)
- ✅ Config changes reflected in behavior
- ✅ Consistent performance across multiple runs
- ✅ Reduced death by starvation/collision
- ✅ Improved survival time and food collection

## Current Status

After fixing config loading:
- ✅ Config system working correctly
- ✅ All 34 parameters loaded from config.yaml
- ⚠️ Current win rate: 0% (default config values need tuning)
- 📝 Parameter optimization needed to reach 47% target

## Troubleshooting

**Config not loading:**
- Check config.yaml syntax (use YAML validator)
- Verify file exists in working directory
- Check server logs for error messages

**Performance worse after changes:**
- Revert to previous config
- Make smaller incremental changes
- Test with more games for statistical significance

**Server not reflecting changes:**
- Ensure server was restarted (kill and restart)
- Verify correct config.yaml being used
- Check logs confirm new values loaded
