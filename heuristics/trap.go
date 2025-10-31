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
