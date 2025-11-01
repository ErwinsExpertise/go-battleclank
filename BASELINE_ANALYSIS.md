# Baseline Snake Analysis

## Overview
The baseline Rust snake (located in `baseline/`) currently defeats our Go snake approximately 100% of the time. This document analyzes its strategy to identify why it wins and how we can counter it.

## Key Strategic Components

### 1. Trap Detection Using Space Ratios
**Implementation**: `baseline/scoring.rs` lines 230-277

The baseline snake uses **space-to-body-length ratios** to detect trap situations:
- **Critical Trap** (< 40% ratio): -600 points
- **Severe Trap** (40-60% ratio): -450 points  
- **Moderate Trap** (60-80% ratio): -250 points
- **Good Space** (80%+ ratio): Safe

**Why it works**:
- Accounts for tail movement (doesn't need 100% of body length in space)
- Early warning system prevents getting boxed in
- Graduated penalties allow measured risk-taking

**Our current approach**: Uses absolute space values without ratio-based thresholds

### 2. Food Death Trap Detection
**Implementation**: `baseline/scoring.rs` lines 310-343

When evaluating food moves, the baseline snake:
1. Simulates eating the food (tail doesn't move)
2. Runs flood fill with `will_grow=true` flag
3. Requires **70% of body length** in remaining space
4. Applies -800 penalty if insufficient space

**Why it works**:
- Prevents fatal food traps where snake boxes itself in after eating
- Different threshold (70%) vs normal movement (40-80%) because tail stays
- Critical safety check that saves lives

**Our current approach**: Basic food danger detection based on enemy proximity

### 3. One-Move Lookahead
**Implementation**: `baseline/scoring.rs` lines 280-307

For each move, the baseline snake:
1. Evaluates current space ratio
2. Simulates all 4 possible next moves
3. Finds worst-case future space ratio
4. If worst future ratio < 80% of current ratio: -200 penalty

**Why it works**:
- Detects dead-end paths early
- Prevents walking into traps
- Lightweight (only 1 move ahead, not full game tree)

**Our current approach**: No lookahead simulation in refactored version

### 4. Aggressive Enemy Pursuit
**Implementation**: `baseline/scoring.rs` lines 204-217

When larger than an enemy within distance 5:
- Distance 2: +100 points (almost in range)
- Distance 3: +50 points (closing in)
- Distance 4: +25 points (still relevant)
- Distance 5: +10 points (on radar)

**Why it works**:
- Actively hunts weaker opponents
- Creates offensive pressure
- Forces enemies into defensive positions

**Our current approach**: Trap detection exists but less aggressive pursuit

### 5. BFS-Based Flood Fill
**Implementation**: `baseline/scoring.rs` lines 446-512

Uses iterative BFS with queue instead of recursion:
- Capped at 121 squares (11×11) for performance
- Properly handles tail movement vs growth
- Consistent with all other calculations

**Why it works**:
- More accurate than depth-limited recursion
- Predictable performance characteristics
- No stack overflow risk

**Our current approach**: Recursive flood fill with depth limits

### 6. Graduated Scoring Weights
**Implementation**: `baseline/scoring.rs` lines 42-79

Well-balanced default weights:
- Instant death: -1000 (absolute)
- Food death trap: -800 (very bad)
- Critical trap: -600 (severe)
- Head-to-head loss: -500 (avoid)
- Severe trap: -450 (bad)
- Trap penalty: -250 (concerning)
- Dead end ahead: -200 (warning)

**Why it works**:
- Clear hierarchy of priorities
- Survival first, then tactics
- Measurable risk vs reward

### 7. Health-Based Food Urgency
**Implementation**: `baseline/scoring.rs` lines 352-364

Dynamic food multipliers:
- Health < 15: 2.0× multiplier + starving bonus (+80)
- Health < 30: 1.5× multiplier + hungry bonus (+40)
- Health ≥ 30: 1.0× multiplier (normal)

**Why it works**:
- Aggressive food seeking when needed
- Doesn't over-prioritize food when healthy
- Balanced survival instinct

## Strategic Weaknesses to Exploit

### 1. Predictable Center Control
The baseline snake has only +2 per unit distance to center. This is relatively weak and can be exploited by:
- Stronger center control early game
- Cutting off enemy's path to center
- Using center position for aggressive plays

### 2. Limited Lookahead Depth
Only looks 1 move ahead. We could:
- Implement 2-move lookahead for tactical advantage
- Set up traps that only become apparent 2+ moves later
- Create false "safe" positions

### 3. Fixed Pursuit Ranges
Pursuit bonuses stop at distance 5. We could:
- Lure enemies beyond distance 5
- Use hit-and-run tactics
- Control engagement distance

### 4. Static Weight System
Weights don't adapt to opponent behavior. We could:
- Learn opponent patterns
- Adjust strategy mid-game
- Exploit predictable responses

## Recommended Counter-Strategies

### Priority 1: Match Core Strengths
1. Implement ratio-based trap detection
2. Add food death trap checks (70% threshold)
3. Add 1-move lookahead with dead-end detection
4. Use BFS-based flood fill

### Priority 2: Exploit Weaknesses
1. Stronger center control (+8-10 instead of +2)
2. 2-move lookahead when computationally feasible
3. Dynamic engagement distance control
4. Adaptive weight adjustments based on game state

### Priority 3: Aggressive Improvements
1. More aggressive pursuit when size advantage exists
2. Better head-to-head positioning
3. Active space denial strategies
4. Food control and denial tactics

## Implementation Roadmap

1. **Phase 1 - Core Parity** (must have)
   - [ ] Ratio-based trap detection
   - [ ] Food death trap detection
   - [ ] 1-move lookahead
   - [ ] BFS flood fill option

2. **Phase 2 - Tactical Edge** (competitive advantage)
   - [ ] Enhanced pursuit logic
   - [ ] 2-move lookahead
   - [ ] Dynamic aggression tuning
   - [ ] Space denial tactics

3. **Phase 3 - Strategic Dominance** (win consistently)
   - [ ] Adaptive weights
   - [ ] Pattern recognition
   - [ ] Food control strategies
   - [ ] Endgame optimization

## Success Metrics

- **Target**: 80%+ win rate vs baseline in 100 game sample
- **Minimum**: 60% win rate (competitive parity)
- **Stretch**: 90%+ win rate (clear dominance)

Metrics to track:
- Win/loss ratio
- Cause of death (starve, head-on, trap, timeout)
- Average survival turns
- Average food collected
- Average final length
- Space control percentage
