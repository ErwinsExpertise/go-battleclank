package main

import (
	"container/heap"
	"log"
	"math"
)

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
	MaxAStarNodes = 200 // Maximum nodes to explore in A* search
)

// info returns metadata about the battlesnake
func info() BattlesnakeInfoResponse {
	log.Println("INFO")
	return BattlesnakeInfoResponse{
		APIVersion: "1",
		Author:     "go-battleclank",
		Color:      "#FF6B35",
		Head:       "trans-rights-scarf",
		Tail:       "bolt",
		Version:    "1.0.0",
	}
}

// start is called when the game begins
func start(state GameState) {
	log.Printf("GAME START: %s\n", state.Game.ID)
}

// end is called when the game finishes
func end(state GameState) {
	log.Printf("GAME OVER: %s\n", state.Game.ID)
}

// move is the main decision-making function called on every turn
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

	// Food seeking
	if state.You.Health < HealthLow {
		foodFactor := evaluateFoodProximity(state, nextPos)
		score += foodFactor * 200.0
	}

	// Avoid smaller snakes' heads (they might kill us in head-to-head)
	headCollisionRisk := evaluateHeadCollisionRisk(state, nextPos)
	score -= headCollisionRisk * 500.0

	// Prefer center positions early game
	if state.Turn < 50 {
		centerFactor := evaluateCenterProximity(state, nextPos)
		score += centerFactor * 10.0
	}

	// Tail chasing when safe
	if state.You.Health > HealthCritical {
		tailFactor := evaluateTailProximity(state, nextPos)
		score += tailFactor * 50.0
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
func evaluateFoodProximity(state GameState, pos Coord) float64 {
	if len(state.Board.Food) == 0 {
		return 0
	}

	// Use A* for critical health situations (more accurate pathfinding)
	if state.You.Health < HealthCritical {
		_, path := findNearestFoodWithAStar(state, pos)
		if path != nil && len(path) > 0 {
			pathLength := len(path)
			if pathLength == 1 {
				return 1.0 // Already at food
			}
			return 1.0 / float64(pathLength)
		}
		// If no path found, fall through to Manhattan distance
	}

	// Use Manhattan distance for non-critical situations (better performance)
	minDist := math.MaxInt32
	for _, food := range state.Board.Food {
		dist := manhattanDistance(pos, food)
		if dist < minDist {
			minDist = dist
		}
	}

	// Inverse distance (closer is better)
	if minDist == 0 {
		return 1.0
	}
	return 1.0 / float64(minDist)
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

	for current != nil {
		path = append([]Coord{current.pos}, path...)
		current = current.parent
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
