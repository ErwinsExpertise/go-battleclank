package heuristics

import "github.com/ErwinsExpertise/go-battleclank/engine/board"

// Package heuristics - danger zone evaluation

// DangerZone represents positions that enemy snakes can reach next turn
type DangerZone map[board.Coord][]board.Snake

// PredictEnemyDangerZones returns all positions that enemy snakes could move to
// Maps each position to the list of enemy snakes that can reach it
func PredictEnemyDangerZones(state *board.GameState) DangerZone {
	dangerZone := make(DangerZone)

	for _, snake := range state.Board.Snakes {
		// Skip our own snake
		if snake.ID == state.You.ID {
			continue
		}

		// For each possible direction the enemy can move
		for _, move := range board.AllMoves() {
			nextPos := board.GetNextPosition(snake.Head, move)

			// Skip if move would be fatal for enemy
			if !state.Board.IsInBounds(nextPos) || state.Board.IsOccupied(nextPos, true) {
				continue
			}

			// Add this position to danger zone
			dangerZone[nextPos] = append(dangerZone[nextPos], snake)
		}
	}

	return dangerZone
}

// IsDangerousPosition checks if a position is in an enemy danger zone
// Returns true if any enemy can move to this position
func IsDangerousPosition(dangerZone DangerZone, pos board.Coord) bool {
	_, exists := dangerZone[pos]
	return exists
}

// GetDangerLevel evaluates how dangerous a position is based on enemy snakes that can reach it
// Returns a danger score: higher = more dangerous
func GetDangerLevel(dangerZone DangerZone, pos board.Coord, ourLength int) float64 {
	enemies, inDanger := dangerZone[pos]
	if !inDanger {
		return 0.0
	}

	danger := 0.0
	for _, enemy := range enemies {
		if enemy.Length > ourLength+1 {
			// Enemy is significantly larger - very dangerous
			danger += 700.0
		} else if enemy.Length >= ourLength {
			// Enemy is same size or slightly larger
			danger += 400.0
		} else {
			// Enemy is smaller - less dangerous
			danger += 100.0
		}
	}

	return danger
}

// IsHeadToHeadRisky evaluates if a position risks head-to-head collision with larger snakes
func IsHeadToHeadRisky(state *board.GameState, pos board.Coord) float64 {
	risk := 0.0
	ourLength := state.You.Length

	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}

		// Check if enemy head could move to adjacent positions
		for _, move := range board.AllMoves() {
			enemyNextPos := board.GetNextPosition(snake.Head, move)

			if enemyNextPos.X == pos.X && enemyNextPos.Y == pos.Y {
				// Potential head-to-head
				if snake.Length >= ourLength {
					risk += 1.0
				} else {
					risk += 0.3
				}
			}
		}
	}

	return risk
}

// DetectCutoff checks if a position has limited escape routes (being boxed in)
func DetectCutoff(state *board.GameState, pos board.Coord) float64 {
	validMoves := 0
	blockedByEnemies := 0

	for _, move := range board.AllMoves() {
		nextPos := board.GetNextPosition(pos, move)

		// Check if valid
		if !state.Board.IsInBounds(nextPos) {
			continue
		}

		// Check if blocked
		blocked := false
		blockedByEnemy := false

		for _, snake := range state.Board.Snakes {
			for i, segment := range snake.Body {
				// Skip tails that will move
				if i == len(snake.Body)-1 && snake.Health != 100 {
					continue
				}
				if nextPos.X == segment.X && nextPos.Y == segment.Y {
					blocked = true
					if snake.ID != state.You.ID {
						blockedByEnemy = true
					}
					break
				}
			}
			if blocked {
				break
			}
		}

		if !blocked {
			validMoves++
		} else if blockedByEnemy {
			blockedByEnemies++
		}
	}

	// Return penalty based on how trapped we are
	if validMoves == 0 {
		return 10.0 // Completely trapped
	} else if validMoves == 1 {
		return 5.0 // Only one escape route
	} else if validMoves == 2 && blockedByEnemies >= 2 {
		return 2.0 // Limited options
	}

	return 0.0
}

// IsBeingChased detects if an enemy snake is following us
func IsBeingChased(state *board.GameState) bool {
	if len(state.You.Body) == 0 {
		return false
	}

	myHead := state.You.Head
	myTail := state.You.Body[len(state.You.Body)-1]

	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}

		// Check if enemy head is close to our tail
		distToTail := board.ManhattanDistance(snake.Head, myTail)
		distToHead := board.ManhattanDistance(snake.Head, myHead)

		// Enemy is chasing if closer to tail than head and within range
		if distToTail <= 3 && distToTail < distToHead {
			return true
		}
	}

	return false
}
