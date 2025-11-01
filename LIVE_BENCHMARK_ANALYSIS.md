# Live Benchmark Analysis - 35% Win Rate

## Current Status

**Live Performance**: 35 wins / 62 losses / 3 errors out of 100 games = **35% win rate**
**Target**: 80% win rate
**Gap**: Need to improve by 45 percentage points

## Key Findings

### Simulation vs Reality Gap

- **Simulated tests**: 96% win rate (using greedy enemy AI)
- **Live tests vs actual baseline**: 35% win rate
- **Root cause**: Simulated enemies use much simpler logic than the actual Rust baseline snake

The baseline Rust snake implements sophisticated trap detection, lookahead, and survival strategies that our simulated tests don't replicate.

## Required Improvements (from BASELINE_ANALYSIS.md)

Based on analysis of the baseline Rust snake's scoring.rs, we need to implement:

### 1. Ratio-Based Trap Detection ⚠️ CRITICAL

**Current approach**: Uses absolute space values
**Baseline approach**: Uses space-to-body-length ratios with graduated penalties

```
Ratios needed:
- Critical Trap (< 40% ratio): -600 penalty
- Severe Trap (40-60% ratio): -450 penalty  
- Moderate Trap (60-80% ratio): -250 penalty
- Good Space (80%+ ratio): Safe
```

**Why it matters**: Accounts for the fact that snakes don't need 100% of body length in space (tail moves). Provides early warning before getting boxed in.

### 2. Food Death Trap Detection ⚠️ CRITICAL

**Current approach**: Basic food danger detection based on enemy proximity
**Baseline approach**: Simulates eating food and checks remaining space

```
Algorithm:
1. Simulate eating the food (tail doesn't move)
2. Run flood fill with will_grow=true flag
3. Require 70% of body length in remaining space
4. Apply -800 penalty if insufficient
```

**Why it matters**: Prevents fatal food traps where snake boxes itself in after eating.

### 3. One-Move Lookahead 📊 HIGH PRIORITY

**Current approach**: Multi-turn lookahead exists but may be too slow
**Baseline approach**: Lightweight 1-move lookahead

```
Algorithm for each move:
1. Evaluate current space ratio
2. Simulate all 4 possible next moves
3. Find worst-case future space ratio
4. If worst future < 80% of current: -200 penalty
```

**Why it matters**: Detects dead-end paths early without expensive multi-turn simulation.

### 4. BFS-Based Flood Fill 📈 MEDIUM PRIORITY

**Current approach**: Recursive flood fill with depth limiting
**Baseline approach**: Iterative BFS with queue

**Why it matters**:
- More accurate than depth-limited recursion
- Predictable performance (capped at 121 squares for 11×11)
- No stack overflow risk
- Consistent with all other calculations

### 5. Graduated Scoring Weights 📈 MEDIUM PRIORITY

**Current weights need review**:
```
Baseline hierarchy:
- Instant death: -1000
- Food death trap: -800
- Critical trap: -600
- Head-to-head loss: -500
- Severe trap: -450
- Moderate trap: -250
- Dead end ahead: -200
```

Our current weights should be aligned with these priorities.

## Implementation Priority

### Phase 1 - Critical Fixes (Target: 60% win rate)
1. ✅ Create live benchmark infrastructure
2. ⬜ Implement ratio-based trap detection (40%, 60%, 80%)
3. ⬜ Implement food death trap detection (70% threshold)
4. ⬜ Run 100-game benchmark to validate

### Phase 2 - High Priority (Target: 75% win rate)
1. ⬜ Implement one-move lookahead
2. ⬜ Align scoring weights with baseline hierarchy
3. ⬜ Run 100-game benchmark to validate

### Phase 3 - Polish (Target: 80%+ win rate)
1. ⬜ Implement BFS flood fill (optional if targets met)
2. ⬜ Fine-tune weights based on failure analysis
3. ⬜ Run 200-game final benchmark

## Success Metrics

- **Milestone 1**: 60% win rate (competitive parity)
- **Milestone 2**: 75% win rate (strong performance)
- **Target**: 80%+ win rate (meets requirements)
- **Stretch**: 90%+ win rate (dominance)

## Next Steps

1. Implement ratio-based trap detection
2. Implement food death trap detection
3. Run iterative benchmarks after each change
4. Document what works and what doesn't
5. Continue improving until 80%+ achieved
