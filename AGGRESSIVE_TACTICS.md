# Aggressive Tactics & Self-Collision Avoidance

This document describes the advanced defensive and offensive strategies implemented to handle aggressive Battlesnake opponents and prevent self-collisions.

## Table of Contents
1. [Overview](#overview)
2. [Self-Collision Prevention](#self-collision-prevention)
3. [Enemy Move Prediction](#enemy-move-prediction)
4. [Danger Zone Detection](#danger-zone-detection)
5. [Cutoff Detection](#cutoff-detection)
6. [Food Safety Enhancement](#food-safety-enhancement)
7. [Anti-Chasing Defense](#anti-chasing-defense)
8. [Dynamic Aggression Adjustment](#dynamic-aggression-adjustment)
9. [Scoring Weights](#scoring-weights)
10. [Performance Considerations](#performance-considerations)

## Overview

This implementation addresses common aggressive Battlesnake tactics:

### Common Aggressive Tactics Countered
- **Head-on Collisions**: Enemy forces head-to-head where they're longer
- **Cutoff/Boxing In**: Enemy blocks escape routes or traps in small areas
- **Food Baiting**: Enemy camps near food to create traps
- **Corner Pressure**: Enemy pushes toward corners or walls
- **Aggressive Chasing**: Enemy follows tail closely to force mistakes

### Defense Strategy
Our approach combines:
1. **Predictive analysis**: Anticipate enemy moves before they happen
2. **Space preservation**: Maintain access to open areas
3. **Risk assessment**: Evaluate danger levels dynamically
4. **Adaptive behavior**: Adjust strategy based on relative strength

## Self-Collision Prevention

### `simulateMove(state, move, turnsAhead) -> bool`

Performs lightweight validation that a move leaves adequate escape routes.

**Purpose**: Prevent snake from turning into itself or getting trapped by its own body.

**Algorithm**:
```
1. Calculate next head position
2. Check if immediately fatal
3. Simulate body shift (head moves, tail leaves)
4. Count available escape routes from new position
5. Return true if ≥1 escape route available
```

**Key Features**:
- Lightweight check (not full recursive simulation)
- Includes food positions for accuracy
- Conservative approach (slightly pessimistic for safety)
- Penalty: -3000 points (allows as last resort)

**Limitations**:
- Doesn't simulate food consumption (assumes tail always moves)
- Doesn't predict enemy movement in simulation
- Only looks 1 turn ahead for performance

**Example Scenario**:
```
Snake at (5,5) heading toward corner at (0,0):
- Head: (5,5) → (4,5) → (3,5) ...
- Body trailing behind

simulateMove() will detect when moving toward (1,0) leaves
only 1-2 escape routes, preventing corner trap.
```

## Enemy Move Prediction

### `predictEnemyMoves(state) -> map[Coord][]Battlesnake`

Calculates all possible next positions for enemy snake heads.

**Purpose**: Identify "danger zones" where enemies can attack next turn.

**Algorithm**:
```
For each enemy snake:
  For each direction (up, down, left, right):
    nextPos = enemy.head + direction
    if not immediately fatal:
      Add nextPos to danger map with enemy reference
```

**Returns**: Map of coordinates → list of enemies that can reach them

**Usage in Scoring**:
- Positions reachable by larger/equal enemies: -800 points
- Positions reachable by smaller enemies: -200 points

**Example**:
```
Our head: (5,5)
Enemy (length 6) head: (7,5)

Predicted enemy moves:
- (7,6) ✓
- (7,4) ✓
- (6,5) ✓ DANGER ZONE (adjacent to us)
- (8,5) ✓

If we move to (6,5), we enter a danger zone!
Score penalty: -800 points
```

## Danger Zone Detection

Integrated into `scoreMove()` via `predictEnemyMoves()`.

**Purpose**: Avoid moving into squares where enemies can attack.

**Danger Levels**:
1. **High Danger** (-800): Larger/equal enemy can reach position
2. **Medium Danger** (-200): Smaller enemy can reach position
3. **Safe** (0): No enemies can reach position

**Decision Logic**:
```go
enemyMoves := predictEnemyMoves(state)
if enemies, inDanger := enemyMoves[nextPos]; inDanger {
    for _, enemy := range enemies {
        if enemy.Length >= state.You.Length {
            score -= 800.0  // High danger
        } else {
            score -= 200.0  // Medium danger
        }
    }
}
```

**Special Cases**:
- Multiple enemies can reach same position → cumulative penalty
- Enemy head-to-head when smaller → we'd win, lower penalty
- Already evaluated in `evaluateHeadCollisionRisk()` → complementary checks

## Cutoff Detection

### `detectCutoff(state, pos) -> float64`

Measures how trapped we are at a given position.

**Purpose**: Detect when enemies are boxing us in or blocking escape routes.

**Algorithm**:
```
1. Count valid moves from position (not blocked, not out of bounds)
2. Count how many are blocked by enemy snakes
3. Return penalty based on available options:
   - 0 valid moves: 10.0 (extreme danger - completely trapped)
   - 1 valid move:  5.0 (high danger - only one escape)
   - 2 valid moves with 2+ blocked by enemies: 2.0 (moderate)
   - 3+ valid moves: 0.0 (safe)
```

**Penalty Application**:
```go
cutoffPenalty := detectCutoff(state, nextPos)
score -= cutoffPenalty * 300.0  // Up to -3000 points
```

**Example Scenarios**:

**Scenario 1: Completely Trapped**
```
Enemy snake forms a box around position (3,3):
  ███
  █.█  ← Our head would be here
  ███
  
Valid moves: 0
Penalty: 10.0 * 300 = -3000 points
```

**Scenario 2: One Escape**
```
Enemy blocks 3 sides:
  ███
  █.░  ← One escape to the right
  █.█
  
Valid moves: 1
Penalty: 5.0 * 300 = -1500 points
```

## Food Safety Enhancement

### `isFoodDangerous(state, food) -> bool`

Enhanced to detect food baiting traps.

**Original Checks**:
1. Food within 2 spaces of enemy body segments

**New Checks**:
2. Enemy can reach food 2+ moves faster than us
3. Enemy reaches within ±1 move and is same/larger size

**Algorithm**:
```go
myDist := manhattanDistance(ourHead, food)
enemyDist := manhattanDistance(enemyHead, food)

// Dangerous if enemy significantly closer
if enemyDist < myDist - 1 {
    return true
}

// Dangerous if equal distance and enemy is larger
if abs(enemyDist - myDist) <= 1 && enemy.Length >= our.Length {
    return true
}
```

**Threshold Rationale**:
- **2+ move difference**: Avoids false positives from incidental proximity
- **±1 with size check**: Head-to-head risk assessment
- **Combined with proximity**: Multi-layer safety check

**Example**:
```
Food at (5,5)
Our head: (0,0) - distance 10
Enemy head: (3,3) - distance 4

Enemy is 6 moves closer (> threshold of 1)
→ Food marked as dangerous
→ Food score multiplied by 0.1 (90% reduction)
```

## Anti-Chasing Defense

### `isBeingChased(state) -> bool`

Detects when enemy snakes are following our tail.

**Purpose**: Identify tail-chasing enemies and take evasive action.

**Detection Logic**:
```go
For each enemy:
    distToOurTail := distance(enemy.head, our.tail)
    distToOurHead := distance(enemy.head, our.head)
    
    if distToOurTail <= 3 && distToOurTail < distToOurHead:
        return true  // Enemy is chasing our tail
```

**Evasive Actions** (when chased):
- Small bonus (+20 points) for moves toward our own body area
- Creates unpredictable movement patterns
- Uses our controlled space where enemy can't easily follow

**Example**:
```
Our snake forming a loop:
  ╔═╗
  ║.║ ← Head
  ╚═╝ ← Tail

Enemy chasing tail:
      ↑
      E

Response: Move toward our own body loop
- Enemy can't follow into our space
- Creates unpredictable path
- Breaks the chase pattern
```

## Dynamic Aggression Adjustment

### `isOutmatchedByNearbyEnemies(state) -> bool`

Detects when we should play defensively vs offensively.

**Purpose**: Adjust strategy based on relative snake sizes.

**Detection Logic**:
```go
For each enemy:
    dist := distance(ourHead, enemyHead)
    if dist <= 3 && enemy.Length > our.Length + 2:
        return true  // Outmatched
```

**Threshold**: 3+ length difference within proximity radius (3 squares)

**Strategy Adjustments When Outmatched**:

| Health Level | Normal Food Weight | Defensive Food Weight | Reduction |
|--------------|-------------------|----------------------|-----------|
| Critical (<30) | 300 | 200 | 33% |
| Low (<50) | 200 | 100 | 50% |
| Healthy (≥50) | 50 | 30 | 40% |

**Space Weight**: Doubles from 100 → 200 when enemies nearby

**Rationale**:
- Reduces risky food-seeking when outmatched
- Prioritizes survival and space over growth
- Still seeks food but more cautiously
- Space becomes more valuable for escape options

**Example**:
```
Our snake: Length 3, Health 40
Enemy nearby: Length 7, Distance 2 squares

Status: Outmatched (length diff = 4, dist = 2)
Food weight: 200 → 100 (50% reduction)
Space weight: 100 → 200 (doubled)

Result: More defensive, space-seeking behavior
```

## Scoring Weights

Complete weight table for all move factors:

| Factor | Base Weight | Condition | Notes |
|--------|-------------|-----------|-------|
| **Fatal Move** | -10000 | Always | Immediate death |
| **Self-Trap** | -3000 | <1 escape route | New feature |
| **Danger Zone** | -800 | Larger enemy can reach | New feature |
| **Danger Zone** | -200 | Smaller enemy can reach | New feature |
| **Cutoff** | -300 per point | Limited escapes | New feature (0-3000) |
| **Head Collision Risk** | -500 per enemy | Enemy can head-to-head | Enhanced |
| **Wall Avoidance** | -400 per penalty | Enemies present | Existing |
| **Space Availability** | +100 | Always | Base weight |
| **Space Availability** | +200 | Enemies nearby | Enhanced (doubled) |
| **Food (Critical)** | +300 | Health < 30, not outmatched | Normal play |
| **Food (Critical)** | +200 | Health < 30, outmatched | Defensive (-33%) |
| **Food (Low)** | +200 | Health < 50, not outmatched | Normal play |
| **Food (Low)** | +100 | Health < 50, outmatched | Defensive (-50%) |
| **Food (Healthy)** | +50 | Health ≥ 50, not outmatched | Normal play |
| **Food (Healthy)** | +30 | Health ≥ 50, outmatched | Defensive (-40%) |
| **Center Proximity** | +10 | Turn < 50 | Early game only |
| **Tail Proximity** | +50 | Health > 30, no nearby enemies | Late game safe |
| **Anti-Chase Bonus** | +20 | Being chased | New feature |

## Performance Considerations

### Time Complexity

| Function | Complexity | Cost | Frequency |
|----------|-----------|------|-----------|
| `predictEnemyMoves()` | O(E × D) | ~1ms | Once per turn |
| `simulateMove()` | O(D × B) | ~2ms | 4 times per turn |
| `detectCutoff()` | O(D × S) | ~1ms | 4 times per turn |
| `isBeingChased()` | O(E) | <1ms | Once per turn |
| `isOutmatchedByNearbyEnemies()` | O(E) | <1ms | Once per turn |

**Legend**:
- E = Number of enemy snakes (typically 1-3)
- D = Directions (always 4)
- B = Body length (typically 3-20)
- S = Total snakes on board (typically 2-4)

**Total Overhead**: ~10-15ms added to decision time
**Original Time**: ~50-100ms for flood-fill and A*
**Total Time**: ~60-115ms (well within 500ms timeout)

### Optimization Strategies

1. **Lightweight Simulation**: `simulateMove()` only validates escape routes, doesn't recursively simulate
2. **Single Pass**: Enemy moves predicted once, reused for all checks
3. **Early Returns**: Functions exit early when danger detected
4. **Simple Distance**: Manhattan distance (O(1)) vs A* pathfinding (O(N log N))
5. **Limited Depth**: No multi-turn recursive simulation

### Memory Usage

- Enemy moves map: O(E × D) ~100 bytes
- Simulated state: O(B × S) ~500 bytes
- Temporary allocations: O(1) per check

**Total Additional Memory**: <1KB (negligible)

## Testing Coverage

### Test Categories

1. **Enemy Move Prediction** (2 tests)
   - Single enemy scenarios
   - Multiple overlapping enemies

2. **Self-Trap Detection** (1 test)
   - Various body configurations
   - Corner and wall scenarios

3. **Outmatched Detection** (3 tests)
   - Large nearby enemy
   - Small nearby enemy
   - Large distant enemy

4. **Chase Detection** (3 tests)
   - Enemy following tail
   - Enemy near head
   - Enemy far away

5. **Cutoff Detection** (3 tests)
   - Open space
   - Completely trapped
   - Limited escape routes

6. **Integration Tests** (4 tests)
   - Danger zone avoidance
   - Defensive play when outmatched
   - Food danger with faster enemy
   - Combined scenarios

**Total New Tests**: 16 focused tests
**Total Coverage**: 91.1% of statements

## Future Enhancements

### Potential Improvements

1. **Multi-Turn Simulation**
   - Extend lookahead to 2-3 turns
   - Account for food consumption
   - Predict enemy movement patterns

2. **Pattern Recognition**
   - Learn common aggressive behaviors
   - Identify opponent strategies
   - Counter known tactics

3. **Coordinated Defense**
   - Work with friendly snakes in team games
   - Share space strategically
   - Coordinate food seeking

4. **Advanced Escape Routes**
   - Use Voronoi diagrams for space control
   - Calculate minimum escape path width
   - Identify chokepoints early

5. **Risk-Based Decisions**
   - Bayesian probability of enemy moves
   - Expected value calculations
   - Monte Carlo simulations for uncertainty

## References

- [Battlesnake Rules](https://docs.battlesnake.com/rules)
- [Battlesnake API](https://docs.battlesnake.com/api)
- [Competitive Strategies Guide](https://docs.battlesnake.com/guides/strategies)
- [A* Pathfinding Implementation](ASTAR_IMPLEMENTATION.md)
- [Overall Strategy Review](STRATEGY_REVIEW.md)
