# Algorithm Documentation

This document provides detailed explanations of the algorithms and optimization strategies used in go-battleclank.

## Table of Contents
1. [Move Scoring System](#move-scoring-system)
2. [Collision Detection](#collision-detection)
3. [Flood Fill Space Analysis](#flood-fill-space-analysis)
4. [Food Seeking Strategy](#food-seeking-strategy)
5. [Head-to-Head Collision Avoidance](#head-to-head-collision-avoidance)
6. [Strategic Positioning](#strategic-positioning)
7. [Performance Optimizations](#performance-optimizations)

## Move Scoring System

The battlesnake evaluates each possible move (up, down, left, right) using a weighted scoring system:

```
Final Score = SpaceScore + FoodScore - CollisionRisk + CenterScore + TailScore
```

### Scoring Weights

| Factor | Weight | Condition |
|--------|--------|-----------|
| Space Availability | 100 | Always |
| Food Proximity | 300 | Health < 30 (critical) |
| Food Proximity | 200 | Health < 50 (low) |
| Food Proximity | 50 | Health ≥ 50 (prevents circling) |
| Head Collision Risk | -500 | Always |
| Center Proximity | 10 | Turn < 50 |
| Tail Proximity | 50 | Health > 30 |
| Fatal Move | -10000 | If move is fatal |

## Collision Detection

### Wall Collision
```go
if pos.X < 0 || pos.X >= width || pos.Y < 0 || pos.Y >= height {
    return fatal
}
```

### Snake Body Collision
The algorithm checks all snake bodies, including our own:

```go
for each snake:
    for each segment (except tail):
        if pos == segment:
            return fatal
```

### Tail Movement Logic
**Key Optimization**: The tail moves unless the snake just ate.

```go
if segment == tail && snake.Health != MaxHealth:
    skip  // Tail will move, safe to occupy
else:
    check collision
```

This allows the snake to follow its own tail when safe.

## Flood Fill Space Analysis

The flood fill algorithm counts reachable spaces from a given position to prevent getting trapped.

### Algorithm
```
floodFill(position, visited, depth):
    if depth > maxDepth or position invalid:
        return 0
    if visited[position]:
        return 0
    if position blocked by snake:
        return 0
    
    visited[position] = true
    count = 1
    
    count += floodFill(position + up, visited, depth+1)
    count += floodFill(position + down, visited, depth+1)
    count += floodFill(position + left, visited, depth+1)
    count += floodFill(position + right, visited, depth+1)
    
    return count
```

### Depth Limiting
To maintain performance under 500ms timeout:
- Max depth = snake length
- Prevents deep recursion on large boards
- Still provides sufficient lookahead for decision making

### Space Score Calculation
```
SpaceScore = (reachable_spaces / total_board_spaces) * 100
```

A higher score indicates more available space, reducing trap risk.

## Food Seeking Strategy

### When to Seek Food
**Always** - Food seeking is now active at all health levels to prevent starvation and circular behavior:
- Health < 30 (critical): Very aggressive food seeking (weight: 300)
- Health < 50 (low): Aggressive food seeking (weight: 200)
- Health ≥ 50 (healthy): Moderate food seeking (weight: 50)

This prevents the snake from going in circles by always maintaining some food awareness.

### Food Score Calculation
```
// For health < 50, use A* pathfinding (accurate around obstacles)
pathLength = aStarSearch(position, nearestFood)
FoodScore = (1 / pathLength) * weight

// For health ≥ 50, use Manhattan distance (faster)
distance = manhattanDistance(position, nearestFood)
FoodScore = (1 / distance) * weight
```

### Manhattan Distance
```
distance = |x1 - x2| + |y1 - y2|
```

This is optimal for grid-based movement where diagonal moves aren't allowed.

### Strategy
1. Always evaluate food proximity (prevents circular tail-chasing)
2. Use A* pathfinding when health < 50 for accurate navigation
3. Use Manhattan distance when healthy for better performance
4. Scale food-seeking weight based on health urgency

## Head-to-Head Collision Avoidance

Head-to-head collisions occur when two snakes move into the same square. The longer snake wins (shorter snake dies).

### Risk Assessment
```go
for each enemy snake:
    for each possible enemy move:
        if enemy_next == our_next:
            if enemy.Length >= our.Length:
                risk += 1.0
```

### Strategy
- Avoid squares adjacent to larger enemy heads
- Only engage if we're bigger
- High negative weight (-500) makes this high priority

### Implementation Detail
Check all 4 possible moves for each enemy head:
```
Enemy at (x, y) could move to:
- (x, y+1)  // up
- (x, y-1)  // down
- (x-1, y)  // left
- (x+1, y)  // right
```

## Strategic Positioning

### Center Preference (Early Game)
In the first 50 turns, prefer center positions:

```
centerScore = 1 - (distance_to_center / max_distance) * 10
```

**Rationale**:
- More food typically spawns near center
- More escape routes available
- Better positioning against opponents

### Tail Chasing (Late Game)
When health > 30, follow own tail:

```
tailScore = (1 / distance_to_tail) * 50
```

**Rationale**:
- Tail position is always safe (it moves)
- Creates a "safe path" to follow
- Maintains position without risk

## Performance Optimizations

### 1. Lazy Evaluation
- Only calculate expensive metrics when needed
- Food proximity only when health < 50
- Center proximity only in first 50 turns

### 2. Early Exit on Fatal Moves
```go
if isImmediatelyFatal(move):
    return -10000
    // Skip all other calculations
```

### 3. Depth-Limited Flood Fill
- Max depth = snake length
- Prevents timeout on large boards
- Still provides good space estimation

### 4. Efficient Data Structures
- Maps for visited tracking (O(1) lookup)
- Pre-calculated head positions
- No unnecessary copying of large structures

### 5. Single-Pass Snake Checks
- One iteration through all snakes
- Check multiple conditions simultaneously
- Minimize redundant iterations

## Time Complexity Analysis

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| isImmediatelyFatal | O(S × B) | S=snakes, B=body length |
| floodFill | O(L × 4^L) | L=snake length (depth limited) |
| foodProximity | O(F) | F=food items |
| headCollisionRisk | O(S) | S=snakes |
| Overall per move | O(S × B + L × 4^L + F) | |

For typical board (11×11, 4 snakes, length ~10):
- Worst case: ~40ms per move
- Well within 500ms timeout
- Leaves room for network latency

## Future Optimization Ideas

> **Note**: For a comprehensive analysis of optimization strategies, see [STRATEGY_REVIEW.md](STRATEGY_REVIEW.md)

### 1. A* Pathfinding
Replace Manhattan distance with A* for food seeking:
- More accurate pathing
- Avoids obstacles
- More computation required
- **Status**: Recommended for Phase 1 implementation
- **Expected Impact**: High - prevents chasing unreachable food

### 2. Minimax Search
Look ahead multiple turns:
- Predict enemy moves
- Evaluate game tree
- Choose optimal path
- High computational cost
- **Status**: Recommended for endgame scenarios
- **Expected Impact**: Medium - tactical advantage in 1v1 situations

### 3. Neural Networks
Train on game history:
- Learn winning patterns
- Adapt to opponent strategies
- Requires training data
- **Status**: Long-term research goal
- **Expected Impact**: Unknown - requires significant investment
- **See**: [STRATEGY_REVIEW.md](STRATEGY_REVIEW.md#deep-q-learning-dqn) for detailed ML analysis

### 4. Monte Carlo Tree Search
Simulate many random games:
- Statistically best move
- Handles uncertainty
- Very effective but slow
- **Status**: Research consideration
- **Expected Impact**: High - proven in similar games, but may exceed time budget

### 5. Voronoi Diagrams
Calculate territory control:
- Identify controlled spaces
- Predict dominance
- Useful for multi-snake scenarios
- **Status**: Lower priority
- **Expected Impact**: Low-Medium - complexity may outweigh benefits

### 6. Genetic Algorithms (NEW)
Optimize heuristic weights automatically:
- Evolve weight configurations through simulated games
- Find counter-intuitive optimal combinations
- Can be tuned for specific opponent types
- **Status**: Recommended for weight tuning
- **Expected Impact**: Medium - improve current heuristics without code changes

### 7. Human-In-the-Loop Learning (NEW)
Combine heuristics with machine learning:
- Use current agent as expert teacher
- Train neural network through behavioral cloning
- Refine with reinforcement learning
- **Status**: Long-term ML approach
- **Expected Impact**: High - faster convergence than pure RL
- **Reference**: https://arxiv.org/pdf/2007.10504.pdf

## Tuning Guidelines

### Increasing Aggression
- Increase food seeking weight (200 → 300)
- Decrease collision avoidance weight (500 → 300)
- Lower health threshold for food (50 → 70)

### Increasing Defense
- Increase space evaluation weight (100 → 150)
- Increase collision avoidance weight (500 → 700)
- Raise health threshold for food (50 → 30)

### Early Game Focus
- Increase center preference weight (10 → 20)
- Extend early game turns (50 → 100)

### Late Game Focus
- Increase tail following weight (50 → 80)
- Decrease center preference duration

## Testing Recommendations

1. **Unit Tests**: Test each heuristic independently
2. **Integration Tests**: Test full move decisions
3. **Simulation**: Play against known strategies
4. **Performance Tests**: Ensure < 500ms response time
5. **Edge Cases**: Test board corners, low health, etc.

## References

### Official Documentation
- [Battlesnake Official Docs](https://docs.battlesnake.com)
- [Battlesnake API Reference](https://docs.battlesnake.com/api)

### Academic Research
- [Battlesnake: A Multi-Agent Reinforcement Learning Testbed](https://arxiv.org/pdf/2007.10504.pdf) - Comprehensive study of ML approaches for Battlesnake
- [Deep Q-Learning (DQN)](https://arxiv.org/abs/1312.5602) - Foundation for RL in games
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - Combining MCTS with neural networks

### Algorithm Fundamentals
- [Flood Fill Algorithm](https://en.wikipedia.org/wiki/Flood_fill)
- [Manhattan Distance](https://en.wikipedia.org/wiki/Taxicab_geometry)
- [Minimax Algorithm](https://en.wikipedia.org/wiki/Minimax)
- [A* Pathfinding](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

### Additional Resources
- [STRATEGY_REVIEW.md](STRATEGY_REVIEW.md) - Comprehensive strategy analysis and implementation roadmap
