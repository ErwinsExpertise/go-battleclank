package heuristics

import (
	"container/heap"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"math"
)

// Package heuristics - food seeking and evaluation

const (
	// DefaultFloodFillDepth is the standard depth limit for flood fill operations
	// This is set below the maximum board cells (11x11 = 121) for performance optimization.
	// Using 93 (approximately 77% of 121) provides sufficient coverage while reducing
	// computation time for pathfinding decisions that need to complete within turn time limits.
	DefaultFloodFillDepth = 93
	
	// Path quality thresholds for space reduction penalties
	// These determine when moving toward food becomes penalized due to space loss
	PathQualityHighReduction    = 0.4  // >40% space reduction gets strong penalty (0.7x multiplier)
	PathQualityModerateReduction = 0.25 // >25% space reduction gets light penalty (0.85x multiplier)
	
	// Path quality multipliers applied to food score
	PathQualityCornerPenalty   = 0.7  // Penalty for food in corners
	PathQualityHighPenalty     = 0.7  // Penalty for high space reduction
	PathQualityModeratePenalty = 0.85 // Penalty for moderate space reduction
	
	// Space availability threshold for corner penalty
	// Only penalize corner food if we have >30% of board space available (not desperate)
	PathQualityCornerSpaceThreshold = 0.3
)

// FindNearestFoodManhattan finds the nearest food using Manhattan distance
func FindNearestFoodManhattan(state *board.GameState, from board.Coord) (board.Coord, int) {
	if len(state.Board.Food) == 0 {
		return board.Coord{}, -1
	}
	
	minDist := math.MaxInt32
	var nearestFood board.Coord
	
	for _, food := range state.Board.Food {
		dist := board.ManhattanDistance(from, food)
		if dist < minDist {
			minDist = dist
			nearestFood = food
		}
	}
	
	return nearestFood, minDist
}

// IsFoodDangerous checks if food is too close to enemy snakes
func IsFoodDangerous(state *board.GameState, food board.Coord, dangerRadius int) bool {
	myDist := board.ManhattanDistance(state.You.Head, food)
	isCritical := state.You.Health < 30
	
	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}
		
		// Check distance to enemy body segments
		for _, segment := range snake.Body {
			dist := board.ManhattanDistance(food, segment)
			if dist <= dangerRadius {
				return true
			}
		}
		
		// Check if enemy can reach significantly faster
		enemyDist := board.ManhattanDistance(snake.Head, food)
		if enemyDist < myDist-2 {
			return true
		}
		
		// Check if arriving at same time with size disadvantage
		lengthAdvantage := snake.Length - state.You.Length
		if abs(enemyDist-myDist) <= 1 {
			if isCritical {
				if lengthAdvantage >= 3 {
					return true
				}
			} else {
				if lengthAdvantage >= 2 {
					return true
				}
			}
		}
	}
	
	return false
}

// abs returns the absolute value of an integer.
// Using a custom implementation instead of math.Abs because math.Abs works with float64,
// and converting int->float64->int adds unnecessary overhead and potential precision issues.
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// EvaluateFoodProximity scores based on distance to nearest safe food
// Now also considers path quality - penalizes paths that lead to worse positions
func EvaluateFoodProximity(state *board.GameState, pos board.Coord, useAStar bool, maxAStarNodes int) float64 {
	if len(state.Board.Food) == 0 {
		return 0
	}
	
	var nearestFood board.Coord
	var distance int
	
	if useAStar && state.You.Health < 50 {
		// Use A* for more accurate pathfinding when health is low
		food, path := FindNearestFoodAStar(state, pos, maxAStarNodes)
		if path != nil && len(path) > 0 {
			nearestFood = food
			distance = len(path)
		} else {
			// Fallback to Manhattan
			nearestFood, distance = FindNearestFoodManhattan(state, pos)
		}
	} else {
		nearestFood, distance = FindNearestFoodManhattan(state, pos)
	}
	
	// Check if food is dangerous
	if IsFoodDangerous(state, nearestFood, 2) {
		// Reduce score for dangerous food
		if distance == 0 {
			return 0.1
		}
		return (1.0 / float64(distance)) * 0.1
	}
	
	// Base food score - inverse of distance (closer food has higher score)
	baseScore := 1.0
	if distance == 0 {
		// Already at food position - maximum score
		baseScore = 1.0
	} else {
		// Score decreases with distance
		baseScore = 1.0 / float64(distance)
	}
	
	// NEW: Evaluate path quality - penalize if moving toward food reduces space/options
	pathQualityFactor := EvaluatePathQuality(state, pos, nearestFood)
	
	return baseScore * pathQualityFactor
}

// EvaluatePathQuality checks if the path toward food leads to progressively worse positions
// Returns a multiplier between 0.5 and 1.0 based on path quality
// Note: This function calls FloodFill twice - once for current position and once for
// the position toward food. Both calls are necessary to compare space availability.
func EvaluatePathQuality(state *board.GameState, fromPos, toFood board.Coord) float64 {
	// Calculate current space at position
	currentSpace := FloodFill(state, fromPos, DefaultFloodFillDepth)
	
	// If we're already very constrained, don't penalize further
	if currentSpace < 10 {
		return 1.0
	}
	
	// Check if food is in a corner or near a wall
	foodDistFromWalls := getMinDistanceFromWalls(state, toFood)
	if foodDistFromWalls == 0 {
		// Food is against a wall - check if it's a corner
		wallCount := 0
		if toFood.X == 0 || toFood.X == state.Board.Width-1 {
			wallCount++
		}
		if toFood.Y == 0 || toFood.Y == state.Board.Height-1 {
			wallCount++
		}
		
		if wallCount >= 2 {
			// Food is in a corner - moderate penalty unless desperate
			totalBoardSpaces := float64(state.Board.Width * state.Board.Height)
			if float64(currentSpace) > PathQualityCornerSpaceThreshold*totalBoardSpaces {
				return PathQualityCornerPenalty
			}
		}
	}
	
	// Calculate direction toward food
	dx := toFood.X - fromPos.X
	dy := toFood.Y - fromPos.Y
	
	// If food is very close (1-2 steps), don't penalize
	manhattanDist := abs(dx) + abs(dy)
	if manhattanDist <= 2 {
		return 1.0
	}
	
	// Normalize to get unit direction
	var moveTowardFood board.Coord
	if abs(dx) > abs(dy) {
		// Move horizontally toward food
		if dx > 0 {
			moveTowardFood = board.Coord{X: fromPos.X + 1, Y: fromPos.Y}
		} else {
			moveTowardFood = board.Coord{X: fromPos.X - 1, Y: fromPos.Y}
		}
	} else {
		// Move vertically toward food
		if dy > 0 {
			moveTowardFood = board.Coord{X: fromPos.X, Y: fromPos.Y + 1}
		} else {
			moveTowardFood = board.Coord{X: fromPos.X, Y: fromPos.Y - 1}
		}
	}
	
	// Check if move toward food is valid
	if !state.Board.IsInBounds(moveTowardFood) || state.Board.IsOccupied(moveTowardFood, true) {
		return 1.0 // Can't move that way anyway, don't apply path quality penalty
	}
	
	// Calculate space at position toward food
	spaceTowardFood := FloodFill(state, moveTowardFood, DefaultFloodFillDepth)
	
	// Calculate space reduction ratio
	spaceReduction := float64(currentSpace-spaceTowardFood) / float64(currentSpace)
	
	// Only penalize if space reduction is significant
	if spaceReduction > PathQualityHighReduction {
		// Moving toward food cuts space by >40% - strong penalty
		return PathQualityHighPenalty
	} else if spaceReduction > PathQualityModerateReduction {
		// Moving toward food cuts space by >25% - light penalty
		return PathQualityModeratePenalty
	}
	
	// Path looks good - no penalty
	return 1.0
}

// getMinDistanceFromWalls returns the minimum distance from any wall
func getMinDistanceFromWalls(state *board.GameState, pos board.Coord) int {
	distFromLeft := pos.X
	distFromRight := state.Board.Width - 1 - pos.X
	distFromBottom := pos.Y
	distFromTop := state.Board.Height - 1 - pos.Y
	
	minDist := distFromLeft
	if distFromRight < minDist {
		minDist = distFromRight
	}
	if distFromBottom < minDist {
		minDist = distFromBottom
	}
	if distFromTop < minDist {
		minDist = distFromTop
	}
	
	return minDist
}

// A* implementation for pathfinding

type aStarNode struct {
	pos    board.Coord
	gScore int
	fScore int
	parent *aStarNode
	index  int
}

type priorityQueue []*aStarNode

func (pq priorityQueue) Len() int           { return len(pq) }
func (pq priorityQueue) Less(i, j int) bool { return pq[i].fScore < pq[j].fScore }
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

// FindNearestFoodAStar finds nearest food using A* pathfinding
func FindNearestFoodAStar(state *board.GameState, start board.Coord, maxNodes int) (board.Coord, []board.Coord) {
	var nearestFood board.Coord
	var shortestPath []board.Coord
	shortestLength := math.MaxInt32
	
	// Try A* to each food
	for _, food := range state.Board.Food {
		path := aStarSearch(state, start, food, maxNodes)
		if path != nil && len(path) < shortestLength {
			shortestPath = path
			nearestFood = food
			shortestLength = len(path)
		}
	}
	
	return nearestFood, shortestPath
}

func aStarSearch(state *board.GameState, start, goal board.Coord, maxNodes int) []board.Coord {
	// Check if goal is blocked
	if state.Board.IsOccupied(goal, true) {
		return nil
	}
	
	if start.X == goal.X && start.Y == goal.Y {
		return []board.Coord{start}
	}
	
	openSet := &priorityQueue{}
	heap.Init(openSet)
	
	startNode := &aStarNode{
		pos:    start,
		gScore: 0,
		fScore: board.ManhattanDistance(start, goal),
		parent: nil,
	}
	heap.Push(openSet, startNode)
	
	visited := make(map[board.Coord]int)
	nodesExplored := 0
	
	for openSet.Len() > 0 && nodesExplored < maxNodes {
		current := heap.Pop(openSet).(*aStarNode)
		nodesExplored++
		
		if current.pos.X == goal.X && current.pos.Y == goal.Y {
			return reconstructPath(current)
		}
		
		visited[current.pos] = current.gScore
		
		for _, neighbor := range state.Board.GetNeighbors(current.pos) {
			if state.Board.IsOccupied(neighbor, true) {
				continue
			}
			
			tentativeGScore := current.gScore + 1
			
			if prevScore, seen := visited[neighbor]; seen && prevScore <= tentativeGScore {
				continue
			}
			
			neighborNode := &aStarNode{
				pos:    neighbor,
				gScore: tentativeGScore,
				fScore: tentativeGScore + board.ManhattanDistance(neighbor, goal),
				parent: current,
			}
			heap.Push(openSet, neighborNode)
		}
	}
	
	return nil
}

func reconstructPath(node *aStarNode) []board.Coord {
	path := []board.Coord{}
	current := node
	
	for current != nil {
		path = append(path, current.pos)
		current = current.parent
	}
	
	// Reverse path
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	
	return path
}
