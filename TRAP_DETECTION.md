# Trap Detection and Aggression Scoring System

This document describes the trap detection and dynamic aggression scoring system that enables the snake to identify and exploit opportunities to trap enemy snakes while maintaining safety.

## Table of Contents
1. [Overview](#overview)
2. [Aggression Scoring System](#aggression-scoring-system)
3. [Trap Detection Logic](#trap-detection-logic)
4. [Reachable Space Comparison](#reachable-space-comparison)
5. [Safety Checks](#safety-checks)
6. [Configuration Constants](#configuration-constants)
7. [Integration with Move Scoring](#integration-with-move-scoring)
8. [Testing](#testing)
9. [Examples](#examples)

## Overview

The trap detection and aggression scoring system implements intelligent offensive behavior by:

- **Calculating dynamic aggression scores** based on health, length, and space control
- **Detecting trap opportunities** where we can box in enemy snakes
- **Validating safety** before committing to aggressive moves
- **Comparing reachable space** for both our snake and enemies

The system only pursues traps when:
1. We have a significant advantage (high aggression score)
2. The trap is safe for us (we maintain sufficient space)
3. We can significantly reduce enemy space (>15% reduction)

## Aggression Scoring System

### Function: `calculateAggressionScore(state GameState) -> float64`

Returns a score from 0.0 (defensive) to 1.0 (aggressive) based on multiple factors.

#### Factors Considered

| Factor | Weight | Description |
|--------|--------|-------------|
| **Health** | ±0.3 | +0.2 when health ≥ 60, -0.3 when health < 30 |
| **Length vs Longest Enemy** | ±0.3 | +0.3 if we're 2+ longer, -0.2 if we're 2+ shorter |
| **Length vs Average** | ±0.1 | +0.1 if we're longer than average |
| **Space Control** | ±0.2 | +0.1 if we control >40% space, -0.2 if <20% |
| **Wall Proximity** | -0.1 | Reduced aggression when close to walls |

#### Scoring Ranges

- **0.0 - 0.3**: Defensive (low health or outmatched)
- **0.4 - 0.6**: Balanced (competitive game state)
- **0.7 - 1.0**: Aggressive (strong advantage)

#### Example Calculations

```go
// Scenario 1: Dominant position
health: 80, length: 10, enemy lengths: [5, 6]
space: 0.5, wall distance: 3
Score: 0.5 + 0.2 (health) + 0.3 (length advantage) + 0.1 (space) = 0.9

// Scenario 2: Defensive position
health: 25, length: 5, enemy lengths: [10, 12]
space: 0.25, wall distance: 1
Score: 0.5 - 0.3 (low health) - 0.2 (outmatched) - 0.1 (wall) = 0.0

// Scenario 3: Balanced
health: 55, length: 8, enemy lengths: [7, 8, 9]
space: 0.35, wall distance: 4
Score: 0.5 + 0.0 (moderate health) + 0.0 (balanced) = 0.5
```

## Trap Detection Logic

### Function: `evaluateTrapOpportunity(state GameState, nextPos Coord) -> float64`

Evaluates whether a move creates a trap opportunity by reducing enemy space.

#### Algorithm

```
For each enemy snake:
  1. Calculate enemy's current reachable space
  2. Simulate our move to nextPos
  3. Calculate enemy's reachable space after our move
  4. Determine space reduction percentage
  
  If space reduction > TrapSpaceThreshold (15%):
    If we maintain safety margin (20% more space):
      If enemy is not much larger:
        Return positive trap score
```

#### Trap Score Calculation

```go
// Base trap score from space reduction
trapScore = spaceReductionPercent

// Multiply by size factor
if enemy.Length <= our.Length:
    trapScore *= 2.0  // Trap smaller/equal snakes aggressively
else if enemy.Length <= our.Length + 2:
    trapScore *= 1.0  // Trap slightly larger snakes cautiously
else:
    trapScore = 0.0   // Don't trap much larger snakes
```

#### Integration Weight

```go
// Only pursue traps when aggression score > 0.6
if aggressionScore > 0.6:
    trapWeight = 200.0 * aggressionScore
    score += trapScore * trapWeight
```

## Reachable Space Comparison

### Function: `evaluateEnemyReachableSpace(state GameState, enemy Battlesnake, fromPos Coord) -> int`

Calculates how many squares an enemy snake can reach using flood fill from a given position.

#### Key Features

- Uses depth-limited flood fill (limited to snake length)
- Excludes the enemy's own tail (which will move)
- Excludes other snakes' tails
- Returns absolute count of reachable squares

### Function: `floodFillForSnake(state GameState, pos Coord, visited map[Coord]bool, depth int, maxDepth int, snakeID string) -> int`

Recursive flood fill algorithm specific to a snake's perspective.

#### Differences from Standard Flood Fill

1. **Snake-specific**: Skips the evaluating snake's tail
2. **Depth limit**: Uses snake length as max depth
3. **No enemy proximity penalty**: Pure reachable space calculation
4. **Performance**: Optimized for trap detection scenarios

## Safety Checks

### Trap Safety Validation

Before pursuing a trap, we verify:

```go
mySimSpace = evaluateSpace(simState, nextPos)
enemySimSpace = float64(enemyNewSpace) / totalSpaces

// Only trap if we have 20% more space
if mySimSpace > enemySimSpace * TrapSafetyMargin:
    // Safe to pursue trap
```

### Safety Margin Rationale

- **1.2x (20% more)**: Conservative but allows aggressive play
- Prevents mutual traps where both snakes get boxed in
- Accounts for enemy movement and board dynamics
- Can be tuned based on performance data

## Configuration Constants

```go
// Trap detection thresholds
TrapSpaceThreshold = 0.15    // 15% minimum space reduction
TrapSafetyMargin = 1.2       // 20% more space than enemy

// Aggression scoring thresholds
AggressionLengthAdvantage = 2  // Length advantage for aggression
AggressionHealthThreshold = 60 // Health needed for aggression
```

### Tuning Guidance

**TrapSpaceThreshold** (currently 0.15):
- Lower (0.10): More trap attempts, may be too aggressive
- Higher (0.20): Fewer traps, more conservative
- Recommended: 0.12-0.18 for balanced play

**TrapSafetyMargin** (currently 1.2):
- Lower (1.1): More aggressive, slightly riskier
- Higher (1.5): Very safe, may miss opportunities
- Recommended: 1.15-1.25 for competitive play

**AggressionHealthThreshold** (currently 60):
- Lower (50): More aggressive at lower health
- Higher (70): More conservative overall
- Recommended: 55-65 based on opponent style

## Integration with Move Scoring

The trap detection and aggression systems integrate into the main `scoreMove` function:

```go
// Calculate aggression score (done once per turn)
aggressionScore := calculateAggressionScore(state)

// Evaluate trap opportunities when aggressive
if aggressionScore > 0.6 {
    trapScore := evaluateTrapOpportunity(state, nextPos)
    trapWeight := 200.0 * aggressionScore
    score += trapScore * trapWeight
}
```

### Scoring Impact

With maximum aggression (1.0) and perfect trap (0.3 space reduction):
```
trapScore = 0.3 * 2.0 = 0.6  (trapping equal-sized enemy)
trapWeight = 200.0 * 1.0 = 200.0
contribution = 0.6 * 200.0 = 120.0 points

This is significant compared to:
- Space evaluation: ~100-200 points
- Food seeking: ~100-400 points
- Wall avoidance: ~300 points
```

## Testing

### Test Coverage

The system includes comprehensive tests in `logic_trap_test.go`:

1. **Aggression Score Tests** (5 scenarios)
   - Healthy and dominant
   - Low health defensive
   - Outmatched defensive
   - Balanced situation
   - Strong position

2. **Trap Detection Tests** (3 scenarios)
   - Basic trap opportunity
   - No trap (enemy far away)
   - Unsafe trap (would endanger us)

3. **Space Calculation Tests**
   - Enemy reachable space
   - Wall distance calculation
   - Game state simulation

4. **Integration Tests**
   - Trap pursuit with advantage
   - Avoid traps when weak

### Running Tests

```bash
# Run all trap-related tests
go test -v -run "Trap|Aggression"

# Run full test suite
go test -v
```

## Examples

### Example 1: Detecting a Trap Opportunity

```
Board State (11x11):
  Our snake (length 7, health 80): dominant position
  Enemy snake (length 3, health 50): in corner
  
  E E .   .   .   .   .   .   .   .   .
  E . .   .   .   .   .   .   .   .   .
  . . .   M   .   .   .   .   .   .   .
  . . M   M   M   .   .   .   .   .   .
  . . .   M   .   .   .   .   .   .   .
  
Calculation:
1. Aggression score = 0.9 (healthy, much longer)
2. Enemy current space = ~30 squares
3. If we move to block: enemy space = ~10 squares
4. Space reduction = 20/121 = 16.5% > 15% threshold
5. Our space after = ~80 squares > 10 * 1.2 = 12 (safe)
6. Trap score = 0.165 * 2.0 = 0.33
7. Contribution = 0.33 * 200 * 0.9 = 59.4 points

Result: TRAP OPPORTUNITY DETECTED
```

### Example 2: Avoiding Unsafe Trap

```
Board State (11x11):
  Our snake (length 5, health 40): near wall
  Enemy snake (length 8, health 70): open space
  
Calculation:
1. Aggression score = 0.2 (moderate health, outmatched)
2. Trap weight threshold not met (0.2 < 0.6)
3. Trap detection not even attempted

Result: PLAYING DEFENSIVELY - no trap attempt
```

### Example 3: Balanced Game State

```
Board State (11x11):
  Our snake (length 8, health 55): mid-board
  Enemy snake (length 8, health 60): mid-board
  
Calculation:
1. Aggression score = 0.5 (balanced)
2. Trap weight threshold not met (0.5 < 0.6)
3. Focus on space control and food seeking

Result: NEUTRAL PLAY - no trap attempt
```

## Performance Considerations

### Computational Cost

| Operation | Complexity | Typical Cost | Frequency |
|-----------|-----------|--------------|-----------|
| calculateAggressionScore | O(E) | ~1ms | Once per turn |
| evaluateTrapOpportunity | O(E × W × H) | ~10-20ms | 4 times per turn |
| evaluateEnemyReachableSpace | O(W × H) | ~3-5ms | Up to 8 times per turn |
| simulateGameState | O(S × B) | <1ms | Up to 4 times per turn |

**Legend:**
- E = Number of enemy snakes (typically 1-3)
- W, H = Board width and height (typically 11x11)
- S = Total snakes (typically 2-4)
- B = Average body length (typically 5-15)

### Total Overhead

- **Without trap detection**: ~50-100ms per turn
- **With trap detection**: ~70-130ms per turn
- **Total time budget**: 500ms (plenty of headroom)

### Optimization Notes

1. **Depth limiting**: Flood fill limited to snake length
2. **Early termination**: Stop if aggression score < 0.6
3. **Caching potential**: Could cache enemy space calculations
4. **Parallel evaluation**: Could evaluate multiple enemies in parallel

## Future Enhancements

### Potential Improvements

1. **Multi-turn trap prediction**
   - Look ahead 2-3 turns for trap setup
   - Predict enemy response to our trap attempts
   - Calculate optimal trap sequences

2. **Collaborative trapping**
   - Coordinate with team snakes in team games
   - Share space control responsibilities
   - Execute pincer movements

3. **Pattern recognition**
   - Learn which opponents fall for traps
   - Identify trap-averse behavior
   - Adapt aggression based on opponent style

4. **Dynamic threshold tuning**
   - Adjust thresholds based on game progression
   - More aggressive in endgame
   - More conservative in early game with many snakes

5. **Escape route analysis**
   - Not just count space, but analyze quality
   - Identify chokepoints and dead ends
   - Weight space by strategic value

## References

- [Main Logic Documentation](AGGRESSIVE_TACTICS.md)
- [Strategy Review](STRATEGY_REVIEW.md)
- [A* Implementation](ASTAR_IMPLEMENTATION.md)
- [Battlesnake API Documentation](https://docs.battlesnake.com/)

## Summary

The trap detection and aggression scoring system provides:

✅ **Dynamic behavior adaptation** based on game state
✅ **Safe aggressive play** with built-in safety checks
✅ **Intelligent trap detection** using space analysis
✅ **Performance-conscious** implementation
✅ **Comprehensive testing** coverage
✅ **Tunable parameters** for different play styles

The system transforms the snake from purely reactive to proactively seeking opportunities to eliminate opponents while maintaining safety as the top priority.
