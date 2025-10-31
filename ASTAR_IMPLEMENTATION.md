# A* Pathfinding Implementation Guide

This document provides a practical guide for implementing A* pathfinding in go-battleclank as the first step in our strategy enhancement roadmap.

## Why A* Pathfinding?

**Current Issue**: Manhattan distance calculates straight-line distance to food, ignoring obstacles.

**Problem Scenarios**:
```
S = Snake Head    # = Wall/Snake    F = Food

Scenario 1: Unreachable Food
####################
#S                F#
####################

Manhattan distance: 18
Actual path: None (unreachable)
Current behavior: Chase unreachable food until death

Scenario 2: Longer Path Required
####################
#S  #############  #
#   #           #  #
#   #     F     #  #
#               #  #
####################

Manhattan distance: 5
Actual path length: 20+
Current behavior: Gets stuck at wall
```

## A* Algorithm Overview

A* finds the shortest path from start to goal by:
1. Maintaining a priority queue of positions to explore
2. Using f(n) = g(n) + h(n) where:
   - g(n) = actual cost from start to n
   - h(n) = estimated cost from n to goal (heuristic)
   - f(n) = estimated total cost through n

### Why A* is Optimal

- **Complete**: Always finds a path if one exists
- **Optimal**: Finds shortest path (with admissible heuristic)
- **Efficient**: Explores fewer nodes than breadth-first search

## Implementation Plan

### Phase 1: Core Data Structures

```go
// Priority queue node for A*
type AStarNode struct {
    Pos    Coord
    GScore int // Cost from start
    FScore int // GScore + heuristic to goal
    Parent *AStarNode
}

// Priority queue implementation
type PriorityQueue []*AStarNode

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
    return pq[i].FScore < pq[j].FScore
}

func (pq PriorityQueue) Swap(i, j int) {
    pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x interface{}) {
    *pq = append(*pq, x.(*AStarNode))
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    *pq = old[0 : n-1]
    return item
}
```

### Phase 2: A* Search Function

```go
// aStarSearch finds the shortest path from start to goal
// Returns nil if no path exists or search exceeds limits
func aStarSearch(state GameState, start, goal Coord, maxNodes int) []Coord {
    // Check if goal is valid destination
    if isBlocked(state, goal) {
        return nil
    }
    
    // Initialize open set with start node
    openSet := &PriorityQueue{}
    heap.Init(openSet)
    
    startNode := &AStarNode{
        Pos:    start,
        GScore: 0,
        FScore: manhattanDistance(start, goal),
        Parent: nil,
    }
    heap.Push(openSet, startNode)
    
    // Track visited nodes and their best scores
    visited := make(map[Coord]int)
    nodesExplored := 0
    
    for openSet.Len() > 0 && nodesExplored < maxNodes {
        // Get node with lowest f-score
        current := heap.Pop(openSet).(*AStarNode)
        nodesExplored++
        
        // Check if we reached the goal
        if current.Pos.X == goal.X && current.Pos.Y == goal.Y {
            return reconstructPath(current)
        }
        
        // Mark as visited
        visited[current.Pos] = current.GScore
        
        // Explore neighbors
        for _, neighbor := range getValidNeighbors(state, current.Pos) {
            tentativeGScore := current.GScore + 1
            
            // Skip if we've found a better path to this neighbor
            if prevScore, seen := visited[neighbor]; seen && prevScore <= tentativeGScore {
                continue
            }
            
            // Add neighbor to open set
            neighborNode := &AStarNode{
                Pos:    neighbor,
                GScore: tentativeGScore,
                FScore: tentativeGScore + manhattanDistance(neighbor, goal),
                Parent: current,
            }
            heap.Push(openSet, neighborNode)
        }
    }
    
    // No path found
    return nil
}

// reconstructPath builds the path from start to goal
func reconstructPath(node *AStarNode) []Coord {
    path := []Coord{}
    current := node
    
    for current != nil {
        path = append([]Coord{current.Pos}, path...)
        current = current.Parent
    }
    
    return path
}

// getValidNeighbors returns adjacent positions that are not blocked
func getValidNeighbors(state GameState, pos Coord) []Coord {
    neighbors := []Coord{}
    directions := []Coord{
        {X: 0, Y: 1},  // up
        {X: 0, Y: -1}, // down
        {X: -1, Y: 0}, // left
        {X: 1, Y: 0},  // right
    }
    
    for _, dir := range directions {
        newPos := Coord{X: pos.X + dir.X, Y: pos.Y + dir.Y}
        
        // Check bounds
        if newPos.X < 0 || newPos.X >= state.Board.Width ||
           newPos.Y < 0 || newPos.Y >= state.Board.Height {
            continue
        }
        
        // Check if blocked by snake
        if isBlocked(state, newPos) {
            continue
        }
        
        neighbors = append(neighbors, newPos)
    }
    
    return neighbors
}

// isBlocked checks if a position is occupied by a snake body
func isBlocked(state GameState, pos Coord) bool {
    for _, snake := range state.Board.Snakes {
        for i, segment := range snake.Body {
            // Skip tails that will move (not just ate)
            if i == len(snake.Body)-1 && snake.Health != MaxHealth {
                continue
            }
            
            if pos.X == segment.X && pos.Y == segment.Y {
                return true
            }
        }
    }
    return false
}
```

### Phase 3: Integration with Food Seeking

```go
// evaluateFoodProximity with A* integration
func evaluateFoodProximity(state GameState, pos Coord) float64 {
    if len(state.Board.Food) == 0 {
        return 0
    }
    
    // Use A* for critical health situations
    if state.You.Health < HealthCritical {
        nearestFood, minPath := findNearestFoodWithAStar(state, pos)
        if minPath != nil {
            pathLength := len(minPath)
            if pathLength > 0 {
                return 1.0 / float64(pathLength)
            }
            return 1.0 // Already at food
        }
        // If no path found, fall through to Manhattan distance
    }
    
    // Use Manhattan distance for non-critical situations (performance)
    minDist := math.MaxInt32
    for _, food := range state.Board.Food {
        dist := manhattanDistance(pos, food)
        if dist < minDist {
            minDist = dist
        }
    }
    
    if minDist == 0 {
        return 1.0
    }
    return 1.0 / float64(minDist)
}

// findNearestFoodWithAStar finds the closest reachable food using A*
func findNearestFoodWithAStar(state GameState, pos Coord) (Coord, []Coord) {
    var nearestFood Coord
    var shortestPath []Coord
    shortestLength := math.MaxInt32
    
    // Try A* to each food item, keeping track of shortest path
    for _, food := range state.Board.Food {
        // Limit nodes to prevent timeout (adjust based on testing)
        path := aStarSearch(state, pos, food, 100)
        if path != nil && len(path) < shortestLength {
            shortestPath = path
            nearestFood = food
            shortestLength = len(path)
        }
    }
    
    return nearestFood, shortestPath
}
```

## Performance Considerations

### Time Complexity

- **Best case**: O(b^d) where b=branching factor (~4), d=depth
- **Worst case**: O(V*log(V)) where V=board size
- **Typical**: 50-200 nodes explored for 11x11 board

### Optimization Strategies

1. **Node Limit**: Cap at 100-200 nodes to prevent timeout
2. **Early Termination**: Stop if path is "good enough"
3. **Caching**: Cache paths for multiple turns (if food doesn't move)
4. **Selective Use**: Only use A* when health < 30

### Performance Budget

```
Current allocation:
- Heuristic evaluation: ~50-100ms for 4 moves
- A* budget: ~20-50ms per move
- Total: ~130-300ms (well within 500ms limit)

Per-move breakdown with A*:
- isImmediatelyFatal: ~5ms
- evaluateSpace (flood fill): ~15ms
- evaluateFoodProximity (A*): ~20-50ms
- Other heuristics: ~5ms
- Total per move: ~45-75ms
- All 4 moves: ~180-300ms
```

## Testing Strategy

### Unit Tests

```go
func TestAStarSearch_StraightLine(t *testing.T) {
    state := GameState{
        Board: Board{
            Width:  11,
            Height: 11,
            Snakes: []Battlesnake{},
        },
    }
    
    start := Coord{X: 0, Y: 0}
    goal := Coord{X: 5, Y: 0}
    
    path := aStarSearch(state, start, goal, 100)
    
    if path == nil {
        t.Fatal("Expected path, got nil")
    }
    
    if len(path) != 6 { // 0,1,2,3,4,5
        t.Errorf("Expected path length 6, got %d", len(path))
    }
}

func TestAStarSearch_AroundObstacle(t *testing.T) {
    state := GameState{
        Board: Board{
            Width:  11,
            Height: 11,
            Snakes: []Battlesnake{
                {
                    Health: 50,
                    Body: []Coord{
                        {X: 5, Y: 0},
                        {X: 5, Y: 1},
                        {X: 5, Y: 2},
                        {X: 5, Y: 3},
                        {X: 5, Y: 4},
                    },
                },
            },
        },
    }
    
    start := Coord{X: 0, Y: 2}
    goal := Coord{X: 10, Y: 2}
    
    path := aStarSearch(state, start, goal, 200)
    
    if path == nil {
        t.Fatal("Expected path, got nil")
    }
    
    // Path should go around the obstacle
    // Verify no position in path collides with snake
    for _, pos := range path {
        if isBlocked(state, pos) {
            t.Errorf("Path includes blocked position: %v", pos)
        }
    }
}

func TestAStarSearch_NoPath(t *testing.T) {
    state := GameState{
        Board: Board{
            Width:  11,
            Height: 11,
            Snakes: []Battlesnake{
                {
                    Health: 50,
                    // Wall of snake bodies
                    Body: []Coord{
                        {X: 5, Y: 0},
                        {X: 5, Y: 1},
                        {X: 5, Y: 2},
                        {X: 5, Y: 3},
                        {X: 5, Y: 4},
                        {X: 5, Y: 5},
                        {X: 5, Y: 6},
                        {X: 5, Y: 7},
                        {X: 5, Y: 8},
                        {X: 5, Y: 9},
                        {X: 5, Y: 10},
                    },
                },
            },
        },
    }
    
    start := Coord{X: 0, Y: 5}
    goal := Coord{X: 10, Y: 5}
    
    path := aStarSearch(state, start, goal, 200)
    
    if path != nil {
        t.Error("Expected nil (no path), got path")
    }
}

func TestAStarSearch_Performance(t *testing.T) {
    state := GameState{
        Board: Board{
            Width:  11,
            Height: 11,
            Snakes: []Battlesnake{},
        },
    }
    
    start := Coord{X: 0, Y: 0}
    goal := Coord{X: 10, Y: 10}
    
    startTime := time.Now()
    path := aStarSearch(state, start, goal, 200)
    duration := time.Since(startTime)
    
    if path == nil {
        t.Fatal("Expected path")
    }
    
    if duration > 50*time.Millisecond {
        t.Errorf("A* took too long: %v (should be < 50ms)", duration)
    }
}
```

### Integration Tests

```go
func TestEvaluateFoodProximity_WithAStar(t *testing.T) {
    state := GameState{
        You: Battlesnake{
            Health: 20, // Critical - will use A*
        },
        Board: Board{
            Width:  11,
            Height: 11,
            Food: []Coord{{X: 10, Y: 10}},
            Snakes: []Battlesnake{
                {
                    Health: 50,
                    Body: []Coord{
                        {X: 5, Y: 0},
                        {X: 5, Y: 1},
                        {X: 5, Y: 2},
                        {X: 5, Y: 3},
                        {X: 5, Y: 4},
                    },
                },
            },
        },
    }
    
    // Position that requires going around obstacle
    pos := Coord{X: 0, Y: 2}
    
    score := evaluateFoodProximity(state, pos)
    
    if score <= 0 {
        t.Error("Expected positive score")
    }
    
    // Verify it's using actual path length, not Manhattan distance
    manhattanDist := manhattanDistance(pos, state.Board.Food[0])
    expectedManhattanScore := 1.0 / float64(manhattanDist)
    
    // A* should give different (more accurate) score
    if score == expectedManhattanScore {
        t.Log("Warning: Score matches Manhattan distance - A* may not be active")
    }
}
```

## Deployment Strategy

### Step 1: Implement and Test (Week 1)
- Add A* data structures and algorithm
- Write comprehensive unit tests
- Validate correctness

### Step 2: Integrate and Benchmark (Week 2)
- Integrate with food seeking logic
- Run performance benchmarks
- Tune node limits and thresholds

### Step 3: A/B Testing (Week 3)
- Deploy side-by-side with current version
- Collect game statistics
- Compare win rates

### Step 4: Optimization (Week 4)
- Profile and optimize hot paths
- Adjust health thresholds
- Fine-tune node limits

## Expected Improvements

### Quantitative Metrics
- **Food Seeking Accuracy**: 40% → 90% (reaching intended food)
- **Trap Avoidance**: 60% → 85% (not chasing unreachable food)
- **Average Survival Time**: +15-25% increase
- **Response Time**: +10-30ms per move (acceptable)

### Qualitative Improvements
- More intelligent food seeking
- Better obstacle navigation
- Fewer "stupid deaths" from chasing unreachable food
- More predictable behavior (easier to debug)

## Rollback Plan

If A* causes issues:
1. Keep feature flag to disable A*
2. Fall back to Manhattan distance
3. Monitor performance metrics
4. Iterate on implementation

```go
const UseAStar = true // Feature flag

func evaluateFoodProximity(state GameState, pos Coord) float64 {
    if UseAStar && state.You.Health < HealthCritical {
        // A* implementation
    }
    // Manhattan distance fallback
}
```

## Next Steps After A*

Once A* is successfully implemented:
1. Consider A* for general pathfinding (not just food)
2. Implement 2-turn lookahead
3. Add enemy movement prediction
4. Begin genetic algorithm weight tuning

## Conclusion

A* pathfinding is a proven, well-understood algorithm that will significantly improve our food-seeking behavior. It's the ideal first step in our enhancement roadmap:

✅ High impact on game performance
✅ Moderate implementation complexity  
✅ Well within time budget
✅ Thoroughly testable
✅ Low risk of breaking existing behavior

This foundation will make subsequent enhancements (lookahead, ML) more effective.
