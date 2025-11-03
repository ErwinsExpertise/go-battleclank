package heuristics

import "github.com/ErwinsExpertise/go-battleclank/engine/board"

// Package heuristics - trap detection and evaluation

const (
	// TrapSpaceThreshold is minimum percentage of space enemy must lose for trap
	TrapSpaceThreshold = 0.15

	// TrapSafetyMargin is minimum space advantage we need when trapping
	TrapSafetyMargin = 1.2
)

// EvaluateTrapOpportunity checks if a move would trap an enemy snake
// Returns a trap score (higher is better) or 0 if not a trap opportunity
func EvaluateTrapOpportunity(state *board.GameState, nextPos board.Coord, maxDepth int) float64 {
	trapScore := 0.0
	totalSpaces := float64(state.Board.Width * state.Board.Height)

	for _, enemy := range state.Board.Snakes {
		if enemy.ID == state.You.ID {
			continue
		}

		// Calculate enemy's current reachable space
		enemyCurrentSpace := FloodFillForSnake(state, enemy.ID, enemy.Head, maxDepth)

		// Simulate the move and check enemy's new space
		// Create a simple simulation by moving our snake
		simState := simulateOurMove(state, nextPos)
		enemyNewSpace := FloodFillForSnake(simState, enemy.ID, enemy.Head, maxDepth)

		// Check space reduction
		spaceReduction := float64(enemyCurrentSpace - enemyNewSpace)
		spaceReductionPercent := spaceReduction / totalSpaces

		// Only consider trap if significant space reduction
		if spaceReductionPercent > TrapSpaceThreshold {
			// Safety check: ensure we have enough space
			mySimSpace := EvaluateSpace(simState, nextPos, maxDepth)
			enemySimSpace := float64(enemyNewSpace) / totalSpaces

			if mySimSpace > enemySimSpace*TrapSafetyMargin {
				// Safe trap opportunity
				if enemy.Length <= state.You.Length {
					trapScore += spaceReductionPercent * 2.0
				} else if enemy.Length <= state.You.Length+2 {
					trapScore += spaceReductionPercent * 1.0
				}
			}
		}
	}

	return trapScore
}

// simulateOurMove creates a simplified game state after our move
func simulateOurMove(state *board.GameState, newHead board.Coord) *board.GameState {
	newState := &board.GameState{
		Turn: state.Turn + 1,
		Board: board.Board{
			Height:  state.Board.Height,
			Width:   state.Board.Width,
			Food:    state.Board.Food,
			Hazards: state.Board.Hazards,
			Snakes:  make([]board.Snake, len(state.Board.Snakes)),
		},
		You: state.You,
	}

	// Copy snakes and update our snake
	for i, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			// Move our snake
			newBody := make([]board.Coord, len(snake.Body))
			newBody[0] = newHead
			copy(newBody[1:], snake.Body[:len(snake.Body)-1])

			newState.Board.Snakes[i] = board.Snake{
				ID:     snake.ID,
				Name:   snake.Name,
				Health: snake.Health,
				Body:   newBody,
				Head:   newHead,
				Length: snake.Length,
			}
			newState.You = newState.Board.Snakes[i]
		} else {
			newState.Board.Snakes[i] = snake
		}
	}

	return newState
}

// IsTrapSafe checks if attempting a trap is safe for us
func IsTrapSafe(state *board.GameState, nextPos board.Coord, maxDepth int) bool {
	simState := simulateOurMove(state, nextPos)
	mySpace := EvaluateSpace(simState, nextPos, maxDepth)

	// Need reasonable amount of space after trap attempt
	return mySpace > 0.15 // At least 15% of board
}

// EvaluateDeadEndAhead performs one-move lookahead to detect dead ends
// Returns penalty if all future moves lead to significantly worse space
// This matches baseline snake's lookahead logic
func EvaluateDeadEndAhead(state *board.GameState, currentPos board.Coord, maxDepth int) float64 {
	// Calculate current space ratio
	currentSpace := FloodFill(state, currentPos, maxDepth)
	bodyLength := state.You.Length
	currentRatio := float64(currentSpace) / float64(bodyLength)

	if currentSpace == 0 {
		return 0.0 // Already trapped, no point in lookahead
	}

	// Simulate one move ahead for each possible direction
	worstNextRatio := currentRatio

	directions := []string{"up", "down", "left", "right"}
	for _, dir := range directions {
		nextPos := getNextPosForDir(currentPos, dir, state)
		if nextPos == nil {
			continue // Invalid move
		}

		// Check if this move is valid
		if !state.Board.IsInBounds(*nextPos) || state.Board.IsOccupied(*nextPos, true) {
			continue
		}

		// Calculate space after this move
		simState := simulateOurMove(state, *nextPos)
		futureSpace := FloodFill(simState, *nextPos, maxDepth)
		futureRatio := float64(futureSpace) / float64(bodyLength)

		if futureRatio < worstNextRatio {
			worstNextRatio = futureRatio
		}
	}

	// If worst future ratio < 80% of current ratio, apply penalty
	// This means all paths lead to significantly worse space
	if worstNextRatio < currentRatio*0.8 {
		return 200.0 // Dead end penalty
	}

	return 0.0
}

// getNextPosForDir returns the next position for a given direction
func getNextPosForDir(pos board.Coord, dir string, state *board.GameState) *board.Coord {
	var nextPos board.Coord
	switch dir {
	case "up":
		nextPos = board.Coord{X: pos.X, Y: pos.Y + 1}
	case "down":
		nextPos = board.Coord{X: pos.X, Y: pos.Y - 1}
	case "left":
		nextPos = board.Coord{X: pos.X - 1, Y: pos.Y}
	case "right":
		nextPos = board.Coord{X: pos.X + 1, Y: pos.Y}
	default:
		return nil
	}
	return &nextPos
}
