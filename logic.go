package main

import (
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
