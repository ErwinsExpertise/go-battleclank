package main

import (
	"container/heap"
	"log"
	"math"
)

// This package implements a STATELESS Battlesnake AI.
// All decision-making functions are pure functions that depend only on the
// current GameState parameter, with no persistent memory between turns.
// This ensures:
// - Updated decisions based on current board state
// - Easy debugging and testing
// - Simple horizontal scaling
// - No accumulation of stale state or bugs
// See: https://medium.com/asymptoticlabs/battlesnake-post-mortem-a5917f9a3428

const (
	// Move directions
	MoveUp    = "up"
	MoveDown  = "down"
	MoveLeft  = "left"
	MoveRight = "right"

	// Health constants
	HealthCritical = 30
	HealthLow      = 50
	MaxHealth      = 100

	// A* algorithm constants
	// MaxAStarNodes limits the number of nodes explored in A* search to prevent timeouts.
	// Tuning guidance:
	// - 11x11 board: 200 nodes (default) - balances accuracy and performance
	// - 7x7 board: 100 nodes - smaller board requires fewer nodes
	// - 19x19 board: 300-400 nodes - larger board may need more exploration
	// Average search explores 50-150 nodes, so 200 provides good headroom.
	MaxAStarNodes = 200

	// Food danger detection constants
	// FoodDangerRadius defines how close (in Manhattan distance) food must be to an enemy
	// snake to be considered dangerous. A radius of 2 provides a safety buffer while not
	// being overly conservative.
	// Tuning guidance:
	// - Radius 1: Very aggressive, only avoids food directly adjacent to enemies
	// - Radius 2: Balanced (default) - avoids traps while still seeking most food
	// - Radius 3: Conservative - may miss food opportunities but maximizes safety
	FoodDangerRadius = 2

	// FoodDangerPenalty is the multiplier applied to food scores when food is dangerous.
	// A penalty of 0.1 means dangerous food is 90% less attractive.
	FoodDangerPenalty = 0.1

	// EnemyProximityRadius defines how close (in Manhattan distance) an enemy snake
	// must be to our head to be considered "nearby". When enemies are within this
	// radius, tail-chasing behavior is disabled to avoid circling near threats.
	// Tuning guidance:
	// - Radius 2: Very cautious - disables circling even with distant enemies
	// - Radius 3: Balanced (default) - reasonable safety zone
	// - Radius 4-5: More relaxed - only disables circling when enemies are very close
	EnemyProximityRadius = 3

	// SpaceBufferRadius defines how close we consider enemy snakes when evaluating space.
	// When flood filling, we treat positions within this radius of enemy heads as blocked
	// to account for the fact that enemies can move and cut off our escape routes.
	// Tuning guidance:
	// - Radius 1: Aggressive - only avoids immediate enemy positions
	// - Radius 2: Balanced (default) - accounts for one enemy move ahead
	// - Radius 3: Conservative - very cautious, may over-restrict movement
	SpaceBufferRadius = 2
)

// info returns metadata about the battlesnake
func info() BattlesnakeInfoResponse {
	log.Println("INFO")
	// Use build version if available, otherwise default
	buildVersion := version
	if buildVersion == "dev" {
		buildVersion = "1.0.0"
	}
	return BattlesnakeInfoResponse{
		APIVersion: "1",
		Author:     "go-battleclank",
		Color:      "#FF6B35",
		Head:       "trans-rights-scarf",
		Tail:       "bolt",
		Version:    buildVersion,
	}
}

// start is called when the game begins
// This function is intentionally stateless - it does not initialize or store
// any game state. All decisions are made fresh on each move based solely on
// the GameState parameter provided by the Battlesnake API.
func start(state GameState) {
	log.Printf("GAME START: %s\n", state.Game.ID)
}

// end is called when the game finishes
// This function is intentionally stateless - it does not clean up or persist
// any game state. This ensures the server remains simple and scalable.
func end(state GameState) {
	log.Printf("GAME OVER: %s\n", state.Game.ID)
}

// move is the main decision-making function called on every turn
// This function is STATELESS and makes decisions based purely on the current
// GameState parameter, ensuring fresh, updated decisions every turn without
// any dependency on previous game history or persistent state.
func move(state GameState) BattlesnakeMoveResponse {
	possibleMoves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}

	// Score each possible move
	moveScores := make(map[string]float64)
	for _, move := range possibleMoves {
		moveScores[move] = scoreMove(state, move)
	}

	// Find the best move
	bestMove := MoveUp
	bestScore := -math.MaxFloat64
	for move, score := range moveScores {
		if score > bestScore {
			bestScore = score
			bestMove = move
		}
	}

	log.Printf("MOVE %d: %s (score: %.2f)\n", state.Turn, bestMove, bestScore)
	return BattlesnakeMoveResponse{
		Move: bestMove,
	}
}

// scoreMove evaluates how good a move is using multiple heuristics
func scoreMove(state GameState, move string) float64 {
	score := 0.0
	myHead := state.You.Head
	nextPos := getNextPosition(myHead, move)

	// Check if move is immediately fatal
	if isImmediatelyFatal(state, nextPos) {
		return -10000.0
	}

	// Space availability (flood fill)
	spaceFactor := evaluateSpace(state, nextPos)
	score += spaceFactor * 100.0

	// Food seeking - always seek food to avoid starvation and circular behavior
	// Weight increases as health decreases
	foodFactor := evaluateFoodProximity(state, nextPos)
	if state.You.Health < HealthCritical {
		// Critical health: aggressive food seeking
		score += foodFactor * 300.0
	} else if state.You.Health < HealthLow {
		// Low health: strong food seeking
		score += foodFactor * 200.0
	} else {
		// Healthy: moderate food seeking to prevent circling
		score += foodFactor * 50.0
	}

	// Avoid smaller snakes' heads (they might kill us in head-to-head)
	headCollisionRisk := evaluateHeadCollisionRisk(state, nextPos)
	score -= headCollisionRisk * 500.0

	// Prefer center positions early game
	if state.Turn < 50 {
		centerFactor := evaluateCenterProximity(state, nextPos)
		score += centerFactor * 10.0
	}

	// Tail chasing when safe - but not when enemies are nearby
	// When enemies are nearby, avoid circling to prevent becoming a sitting target
	if state.You.Health > HealthCritical && !hasEnemiesNearby(state) {
		tailFactor := evaluateTailProximity(state, nextPos)
		score += tailFactor * 50.0
	}

	// Penalize corner/edge positions when enemies are nearby
	// This prevents getting squeezed into corners with limited escape routes
	if hasEnemiesNearby(state) {
		cornerPenalty := evaluateCornerPenalty(state, nextPos)
		score -= cornerPenalty * 150.0
	}

	return score
}

// isImmediatelyFatal checks if a position results in immediate death
func isImmediatelyFatal(state GameState, pos Coord) bool {
	// Out of bounds
	if pos.X < 0 || pos.X >= state.Board.Width || pos.Y < 0 || pos.Y >= state.Board.Height {
		return true
	}

	// Collision with any snake body (including our own)
	for _, snake := range state.Board.Snakes {
		for i, segment := range snake.Body {
			// Skip the tail unless the snake just ate (length will grow)
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

// evaluateSpace uses flood fill to determine available space
func evaluateSpace(state GameState, pos Coord) float64 {
	visited := make(map[Coord]bool)
	count := floodFill(state, pos, visited, 0, state.You.Length)

	// Normalize by board size
	totalSpaces := state.Board.Width * state.Board.Height
	return float64(count) / float64(totalSpaces)
}

// floodFill recursively counts reachable spaces
func floodFill(state GameState, pos Coord, visited map[Coord]bool, depth int, maxDepth int) int {
	// Limit recursion depth for performance
	if depth > maxDepth {
		return 0
	}

	// Check if position is valid and unvisited
	if pos.X < 0 || pos.X >= state.Board.Width || pos.Y < 0 || pos.Y >= state.Board.Height {
		return 0
	}
	if visited[pos] {
		return 0
	}

	// Check if blocked by snake
	for _, snake := range state.Board.Snakes {
		for i, segment := range snake.Body {
			// Skip tails that will move
			if i == len(snake.Body)-1 {
				continue
			}
			if pos.X == segment.X && pos.Y == segment.Y {
				return 0
			}
		}
	}

	// Check if too close to enemy snake heads (they can move and cut us off)
	// Skip our own snake to avoid penalizing our own position
	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}
		// Consider positions near enemy heads as risky during space evaluation
		// Only apply this at shallow depths to avoid over-restricting distant areas
		if depth < 3 {
			dist := manhattanDistance(pos, snake.Head)
			if dist <= SpaceBufferRadius {
				// Don't completely block, but reduce the value of this space
				// by treating it as less valuable (we still count it but flag it)
				visited[pos] = true
				// Return reduced count (0.3 instead of 1.0) for risky positions
				count := 0
				count += floodFill(state, Coord{X: pos.X + 1, Y: pos.Y}, visited, depth+1, maxDepth)
				count += floodFill(state, Coord{X: pos.X - 1, Y: pos.Y}, visited, depth+1, maxDepth)
				count += floodFill(state, Coord{X: pos.X, Y: pos.Y + 1}, visited, depth+1, maxDepth)
				count += floodFill(state, Coord{X: pos.X, Y: pos.Y - 1}, visited, depth+1, maxDepth)
				// Return partial credit for this risky position but continue exploring
				return count / 3
			}
		}
	}

	visited[pos] = true
	count := 1

	// Recursively check adjacent squares
	count += floodFill(state, Coord{X: pos.X + 1, Y: pos.Y}, visited, depth+1, maxDepth)
	count += floodFill(state, Coord{X: pos.X - 1, Y: pos.Y}, visited, depth+1, maxDepth)
	count += floodFill(state, Coord{X: pos.X, Y: pos.Y + 1}, visited, depth+1, maxDepth)
	count += floodFill(state, Coord{X: pos.X, Y: pos.Y - 1}, visited, depth+1, maxDepth)

	return count
}

// evaluateFoodProximity scores based on proximity to nearest food
// Now also considers if food is dangerous due to nearby enemy snakes
func evaluateFoodProximity(state GameState, pos Coord) float64 {
	if len(state.Board.Food) == 0 {
		return 0
	}

	var targetFood Coord
	var distanceScore float64

	// Use A* for better pathfinding when health is low (more accurate around obstacles)
	if state.You.Health < HealthLow {
		food, path := findNearestFoodWithAStar(state, pos)
		if path != nil && len(path) > 0 {
			targetFood = food
			pathLength := len(path)
			if pathLength == 1 {
				distanceScore = 1.0 // Already at food
			} else {
				distanceScore = 1.0 / float64(pathLength)
			}
		} else {
			// If no path found with A*, fall back to Manhattan distance
			targetFood, distanceScore = findNearestFoodManhattan(state, pos)
		}
	} else {
		// Use Manhattan distance for non-critical situations (better performance)
		targetFood, distanceScore = findNearestFoodManhattan(state, pos)
	}

	// Check if the target food is dangerous due to nearby enemy snakes
	if isFoodDangerous(state, targetFood) {
		// Significantly reduce food attractiveness if it's near enemy snakes
		// This prevents the snake from getting trapped near enemy snakes
		distanceScore *= FoodDangerPenalty
	}

	return distanceScore
}

// findNearestFoodManhattan finds the nearest food using Manhattan distance
func findNearestFoodManhattan(state GameState, pos Coord) (Coord, float64) {
	minDist := math.MaxInt32
	var nearestFood Coord

	for _, food := range state.Board.Food {
		dist := manhattanDistance(pos, food)
		if dist < minDist {
			minDist = dist
			nearestFood = food
		}
	}

	// Inverse distance (closer is better)
	distanceScore := 1.0
	if minDist == 0 {
		distanceScore = 1.0
	} else {
		distanceScore = 1.0 / float64(minDist)
	}

	return nearestFood, distanceScore
}

// isFoodDangerous checks if food is too close to enemy snakes
// Food is considered dangerous if it's within FoodDangerRadius spaces of any enemy snake body segment
func isFoodDangerous(state GameState, food Coord) bool {
	for _, snake := range state.Board.Snakes {
		// Skip our own snake
		if snake.ID == state.You.ID {
			continue
		}

		// Check distance to each segment of enemy snake
		for _, segment := range snake.Body {
			dist := manhattanDistance(food, segment)
			if dist <= FoodDangerRadius {
				return true
			}
		}
	}

	return false
}

// evaluateHeadCollisionRisk checks for potential head-to-head collisions
func evaluateHeadCollisionRisk(state GameState, pos Coord) float64 {
	risk := 0.0
	myLength := state.You.Length

	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}

		// Check if enemy snake's head could move to adjacent squares
		enemyHead := snake.Head
		for _, dir := range []string{MoveUp, MoveDown, MoveLeft, MoveRight} {
			enemyNextPos := getNextPosition(enemyHead, dir)
			if enemyNextPos.X == pos.X && enemyNextPos.Y == pos.Y {
				// Head-to-head collision possible
				if snake.Length >= myLength {
					risk += 1.0 // Enemy is same size or larger
				}
			}
		}
	}

	return risk
}

// evaluateCenterProximity prefers center positions
func evaluateCenterProximity(state GameState, pos Coord) float64 {
	centerX := state.Board.Width / 2
	centerY := state.Board.Height / 2
	dist := manhattanDistance(pos, Coord{X: centerX, Y: centerY})
	maxDist := centerX + centerY

	if maxDist == 0 {
		return 1.0
	}
	return 1.0 - (float64(dist) / float64(maxDist))
}

// evaluateTailProximity encourages following own tail when safe
func evaluateTailProximity(state GameState, pos Coord) float64 {
	if len(state.You.Body) < 2 {
		return 0
	}

	tail := state.You.Body[len(state.You.Body)-1]
	dist := manhattanDistance(pos, tail)

	if dist == 0 {
		return 1.0
	}
	return 1.0 / float64(dist)
}

// evaluateCornerPenalty returns a penalty for positions near corners/edges
// when enemies are nearby. This prevents getting squeezed into corners.
func evaluateCornerPenalty(state GameState, pos Coord) float64 {
	// Calculate distance from edges
	distFromLeft := pos.X
	distFromRight := state.Board.Width - 1 - pos.X
	distFromBottom := pos.Y
	distFromTop := state.Board.Height - 1 - pos.Y

	// Find minimum distance to any edge
	minDistToEdge := distFromLeft
	if distFromRight < minDistToEdge {
		minDistToEdge = distFromRight
	}
	if distFromBottom < minDistToEdge {
		minDistToEdge = distFromBottom
	}
	if distFromTop < minDistToEdge {
		minDistToEdge = distFromTop
	}

	// Count how many directions are blocked (by walls or approaching edge)
	blockedDirections := 0
	
	// Check if near walls (within 1-2 squares)
	if distFromLeft <= 1 {
		blockedDirections++
	}
	if distFromRight <= 1 {
		blockedDirections++
	}
	if distFromBottom <= 1 {
		blockedDirections++
	}
	if distFromTop <= 1 {
		blockedDirections++
	}

	// Calculate penalty based on how cornered we are
	// 0 blocked directions = no penalty
	// 1 blocked direction = slight penalty (edge)
	// 2+ blocked directions = high penalty (corner or very limited escape)
	penalty := 0.0
	switch blockedDirections {
	case 0:
		penalty = 0.0
	case 1:
		penalty = 0.3 // Slight penalty for being near an edge
	case 2:
		penalty = 0.8 // High penalty for corner
	default:
		penalty = 1.0 // Maximum penalty for severely cornered position
	}

	// Also factor in overall distance from edges - being far from edges is safer
	if minDistToEdge <= 1 {
		penalty += 0.2
	}

	return penalty
}

// manhattanDistance calculates the Manhattan distance between two coordinates
func manhattanDistance(a, b Coord) int {
	return abs(a.X-b.X) + abs(a.Y-b.Y)
}

// abs returns the absolute value of an integer
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// hasEnemiesNearby checks if any enemy snakes are within EnemyProximityRadius
// of our head. Returns true if enemies are nearby, indicating we should avoid
// circling/tail-chasing behavior to prevent becoming a sitting target.
func hasEnemiesNearby(state GameState) bool {
	myHead := state.You.Head

	for _, snake := range state.Board.Snakes {
		// Skip our own snake
		if snake.ID == state.You.ID {
			continue
		}

		// Check if enemy head is within proximity radius
		dist := manhattanDistance(myHead, snake.Head)
		if dist <= EnemyProximityRadius {
			return true
		}
	}

	return false
}

// getNextPosition returns the coordinate resulting from a move
func getNextPosition(pos Coord, move string) Coord {
	switch move {
	case MoveUp:
		return Coord{X: pos.X, Y: pos.Y + 1}
	case MoveDown:
		return Coord{X: pos.X, Y: pos.Y - 1}
	case MoveLeft:
		return Coord{X: pos.X - 1, Y: pos.Y}
	case MoveRight:
		return Coord{X: pos.X + 1, Y: pos.Y}
	}
	return pos
}

// A* Pathfinding Implementation

// aStarNode represents a node in the A* search
type aStarNode struct {
	pos    Coord
	gScore int // Cost from start
	fScore int // gScore + heuristic to goal
	parent *aStarNode
	index  int // Index in the priority queue
}

// priorityQueue implements heap.Interface for A* algorithm
type priorityQueue []*aStarNode

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	return pq[i].fScore < pq[j].fScore
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueue) Push(x interface{}) {
	n := len(*pq)
	node := x.(*aStarNode)
	node.index = n
	*pq = append(*pq, node)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	node := old[n-1]
	old[n-1] = nil
	node.index = -1
	*pq = old[0 : n-1]
	return node
}

// aStarSearch finds the shortest path from start to goal using A* algorithm
// Returns nil if no path exists or search exceeds maxNodes
func aStarSearch(state GameState, start, goal Coord, maxNodes int) []Coord {
	// Check if goal is blocked
	if isPositionBlocked(state, goal) {
		return nil
	}

	// Check if start equals goal
	if start.X == goal.X && start.Y == goal.Y {
		return []Coord{start}
	}

	// Initialize open set with start node
	openSet := &priorityQueue{}
	heap.Init(openSet)

	startNode := &aStarNode{
		pos:    start,
		gScore: 0,
		fScore: manhattanDistance(start, goal),
		parent: nil,
	}
	heap.Push(openSet, startNode)

	// Track visited nodes and their best scores
	visited := make(map[Coord]int)
	nodesExplored := 0

	for openSet.Len() > 0 && nodesExplored < maxNodes {
		// Get node with lowest f-score
		current := heap.Pop(openSet).(*aStarNode)
		nodesExplored++

		// Check if we reached the goal
		if current.pos.X == goal.X && current.pos.Y == goal.Y {
			return reconstructPath(current)
		}

		// Mark as visited
		visited[current.pos] = current.gScore

		// Explore neighbors
		for _, neighbor := range getValidNeighbors(state, current.pos) {
			tentativeGScore := current.gScore + 1

			// Skip if we've found a better path to this neighbor
			if prevScore, seen := visited[neighbor]; seen && prevScore <= tentativeGScore {
				continue
			}

			// Add neighbor to open set
			neighborNode := &aStarNode{
				pos:    neighbor,
				gScore: tentativeGScore,
				fScore: tentativeGScore + manhattanDistance(neighbor, goal),
				parent: current,
			}
			heap.Push(openSet, neighborNode)
		}
	}

	// No path found
	return nil
}

// reconstructPath builds the path from start to goal by following parent pointers
func reconstructPath(node *aStarNode) []Coord {
	path := []Coord{}
	current := node

	// Build path in reverse (goal to start)
	for current != nil {
		path = append(path, current.pos)
		current = current.parent
	}

	// Reverse the path to get start to goal order
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
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
		if isPositionBlocked(state, newPos) {
			continue
		}

		neighbors = append(neighbors, newPos)
	}

	return neighbors
}

// isPositionBlocked checks if a position is occupied by a snake body
func isPositionBlocked(state GameState, pos Coord) bool {
	for _, snake := range state.Board.Snakes {
		for i, segment := range snake.Body {
			// Skip tails that will move (snake hasn't just eaten)
			// When a snake eats food, its health is reset to MaxHealth and
			// the tail doesn't move on that turn (body grows). In all other
			// cases, the tail moves forward so that position will be empty.
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

// findNearestFoodWithAStar finds the closest reachable food using A*
func findNearestFoodWithAStar(state GameState, pos Coord) (Coord, []Coord) {
	var nearestFood Coord
	var shortestPath []Coord
	shortestLength := math.MaxInt32

	// Try A* to each food item, keeping track of shortest path
	for _, food := range state.Board.Food {
		path := aStarSearch(state, pos, food, MaxAStarNodes)
		if path != nil && len(path) < shortestLength {
			shortestPath = path
			nearestFood = food
			shortestLength = len(path)
		}
	}

	return nearestFood, shortestPath
}
