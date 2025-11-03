package heuristics

import (
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

// Starvation strategy: Cut off opponent's access to food and trap them in a smaller area

// EvaluateStarvationOpportunity detects if we can cut off the opponent from food
// Returns a score indicating how effectively we can starve the opponent
func EvaluateStarvationOpportunity(state *board.GameState, nextPos board.Coord, maxDepth int) float64 {
	score := 0.0

	// Only pursue starvation if we have a health and length advantage
	if state.You.Health < 50 {
		return 0.0 // Don't try starvation if we're hungry
	}

	// Find closest enemy
	var closestEnemy *board.Snake
	minDist := 1000
	myHead := state.You.Head

	for i := range state.Board.Snakes {
		snake := &state.Board.Snakes[i]
		if snake.ID == state.You.ID {
			continue
		}
		dist := board.ManhattanDistance(myHead, snake.Head)
		if dist < minDist {
			minDist = dist
			closestEnemy = snake
		}
	}

	if closestEnemy == nil {
		return 0.0
	}

	// Check if we're longer (required for starvation strategy)
	if state.You.Length <= closestEnemy.Length {
		return 0.0
	}

	// Check if we can create a wall between enemy and food
	wallBonus := evaluateWallFormation(state, nextPos, closestEnemy)
	score += wallBonus

	// Check if we're cutting off enemy's space
	spaceCutoffBonus := evaluateSpaceCutoff(state, nextPos, closestEnemy, maxDepth)
	score += spaceCutoffBonus

	// Bonus for positioning between enemy and food
	foodInterceptionBonus := evaluateFoodInterception(state, nextPos, closestEnemy)
	score += foodInterceptionBonus

	return score
}

// evaluateWallFormation checks if moving to nextPos helps create a wall/barrier
func evaluateWallFormation(state *board.GameState, nextPos board.Coord, enemy *board.Snake) float64 {
	score := 0.0

	// Check if nextPos is on a line between enemy and board edges
	// This helps create a "wall" using our body + board edges

	enemyX := enemy.Head.X
	enemyY := enemy.Head.Y

	// Vertical wall formation (use board edges)
	if nextPos.X == enemyX-1 || nextPos.X == enemyX+1 {
		// We're creating a vertical wall
		score += 50.0

		// Bonus if enemy is near edge (easier to trap)
		if enemyX <= 2 || enemyX >= state.Board.Width-3 {
			score += 50.0
		}
	}

	// Horizontal wall formation
	if nextPos.Y == enemyY-1 || nextPos.Y == enemyY+1 {
		// We're creating a horizontal wall
		score += 50.0

		// Bonus if enemy is near edge
		if enemyY <= 2 || enemyY >= state.Board.Height-3 {
			score += 50.0
		}
	}

	// Bonus if we're positioning near a corner (easier to trap)
	if isNearCorner(state, enemyX, enemyY, 3) {
		score += 100.0
	}

	return score
}

// evaluateSpaceCutoff checks if we're reducing enemy's available space
func evaluateSpaceCutoff(state *board.GameState, nextPos board.Coord, enemy *board.Snake, maxDepth int) float64 {
	// Calculate enemy's space with and without our move

	// Calculate enemy's current accessible space
	enemySpace := EvaluateSpaceFromPosition(state, enemy.Head, maxDepth)

	// Compare to enemy's body length - if space < body * 2, they're constrained
	enemyBodySize := float64(enemy.Length)
	spaceRatio := enemySpace / enemyBodySize

	// Check if our move to nextPos reduces enemy's options
	// (nextPos is closer to enemy and blocks escape routes)
	distToEnemy := board.ManhattanDistance(nextPos, enemy.Head)

	score := 0.0

	// Bonus for being close to enemy when they're constrained
	if spaceRatio < 1.5 && distToEnemy <= 3 {
		// Enemy is severely constrained and we're nearby
		score += 200.0
	} else if spaceRatio < 2.5 && distToEnemy <= 4 {
		// Enemy is moderately constrained
		score += 100.0
	} else if spaceRatio < 4.0 && distToEnemy <= 5 {
		// Enemy is somewhat constrained
		score += 50.0
	}

	return score
}

// evaluateFoodInterception checks if we're positioning between enemy and nearest food
func evaluateFoodInterception(state *board.GameState, nextPos board.Coord, enemy *board.Snake) float64 {
	if len(state.Board.Food) == 0 {
		return 0.0
	}

	score := 0.0

	// Find food closest to enemy
	var closestFood board.Coord
	minDist := 1000

	for _, food := range state.Board.Food {
		dist := board.ManhattanDistance(enemy.Head, food)
		if dist < minDist {
			minDist = dist
			closestFood = food
		}
	}

	// Check if we're between enemy and food
	enemyToFood := board.ManhattanDistance(enemy.Head, closestFood)
	ourToFood := board.ManhattanDistance(nextPos, closestFood)
	enemyToUs := board.ManhattanDistance(enemy.Head, nextPos)

	// If we're closer to food AND between enemy and food
	if ourToFood < enemyToFood {
		// We're closer to food
		score += 30.0

		// Extra bonus if we're directly between them (triangle inequality)
		if enemyToUs+ourToFood <= enemyToFood+2 {
			score += 70.0
		}
	}

	// Bonus for blocking multiple food sources
	foodBlocked := 0
	for _, food := range state.Board.Food {
		ourDist := board.ManhattanDistance(nextPos, food)
		enemyDist := board.ManhattanDistance(enemy.Head, food)

		if ourDist < enemyDist {
			foodBlocked++
		}
	}

	if foodBlocked >= 2 {
		score += float64(foodBlocked) * 40.0
	}

	return score
}

// isNearCorner checks if position is near a corner
func isNearCorner(state *board.GameState, x, y, radius int) bool {
	width := state.Board.Width
	height := state.Board.Height

	// Check all four corners
	corners := []board.Coord{
		{X: 0, Y: 0},
		{X: width - 1, Y: 0},
		{X: 0, Y: height - 1},
		{X: width - 1, Y: height - 1},
	}

	for _, corner := range corners {
		dist := board.ManhattanDistance(board.Coord{X: x, Y: y}, corner)
		if dist <= radius {
			return true
		}
	}

	return false
}

// EvaluateSpaceFromPosition calculates accessible space from a specific position
func EvaluateSpaceFromPosition(state *board.GameState, start board.Coord, maxDepth int) float64 {
	visited := make(map[board.Coord]bool)
	count := floodFillFrom(state, start, visited, 0, maxDepth)
	return float64(count)
}

// floodFillFrom performs flood fill from a specific position
func floodFillFrom(state *board.GameState, pos board.Coord, visited map[board.Coord]bool, depth, maxDepth int) int {
	if depth > maxDepth {
		return 0
	}

	if visited[pos] {
		return 0
	}

	if !state.Board.IsInBounds(pos) || state.Board.IsOccupied(pos, true) {
		return 0
	}

	visited[pos] = true
	count := 1

	for _, move := range board.AllMoves() {
		nextPos := board.GetNextPosition(pos, move)
		count += floodFillFrom(state, nextPos, visited, depth+1, maxDepth)
	}

	return count
}
