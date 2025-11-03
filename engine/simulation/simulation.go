package simulation

import "github.com/ErwinsExpertise/go-battleclank/engine/board"

// Package simulation provides fast game-state copy and step simulation for lookahead

// SimulateMove creates a new game state after applying a move for a specific snake
// This creates a lightweight copy for lookahead purposes
func SimulateMove(state *board.GameState, snakeID string, move string) *board.GameState {
	// Create new state
	newState := &board.GameState{
		Turn: state.Turn + 1,
		Board: board.Board{
			Height:  state.Board.Height,
			Width:   state.Board.Width,
			Food:    make([]board.Coord, len(state.Board.Food)),
			Hazards: make([]board.Coord, len(state.Board.Hazards)),
			Snakes:  make([]board.Snake, len(state.Board.Snakes)),
		},
		You: state.You,
	}

	// Copy food and hazards
	copy(newState.Board.Food, state.Board.Food)
	copy(newState.Board.Hazards, state.Board.Hazards)

	// Copy and update snakes
	for i, snake := range state.Board.Snakes {
		if snake.ID == snakeID {
			// Apply move to this snake
			newHead := board.GetNextPosition(snake.Head, move)

			// Check if snake ate food
			ateFood := false
			for j, food := range newState.Board.Food {
				if newHead.X == food.X && newHead.Y == food.Y {
					ateFood = true
					// Remove food
					newState.Board.Food = append(newState.Board.Food[:j], newState.Board.Food[j+1:]...)
					break
				}
			}

			// Create new body
			newBody := make([]board.Coord, len(snake.Body))
			newBody[0] = newHead

			if ateFood {
				// Snake grows - copy entire body
				copy(newBody[1:], snake.Body)
				newState.Board.Snakes[i] = board.Snake{
					ID:     snake.ID,
					Name:   snake.Name,
					Health: 100, // Reset health after eating
					Body:   newBody,
					Head:   newHead,
					Length: snake.Length + 1,
				}
			} else {
				// Snake doesn't grow - shift body (tail moves)
				copy(newBody[1:], snake.Body[:len(snake.Body)-1])
				newState.Board.Snakes[i] = board.Snake{
					ID:     snake.ID,
					Name:   snake.Name,
					Health: snake.Health - 1, // Decrease health
					Body:   newBody,
					Head:   newHead,
					Length: snake.Length,
				}
			}

			// Update You if this is our snake
			if snake.ID == state.You.ID {
				newState.You = newState.Board.Snakes[i]
			}
		} else {
			// Copy other snakes unchanged
			newState.Board.Snakes[i] = snake
		}
	}

	return newState
}

// IsMoveValid checks if a move is valid (doesn't result in immediate collision)
func IsMoveValid(state *board.GameState, snakeID string, move string) bool {
	snake := state.Board.GetSnakeByID(snakeID)
	if snake == nil {
		return false
	}

	nextPos := board.GetNextPosition(snake.Head, move)

	// Check bounds
	if !state.Board.IsInBounds(nextPos) {
		return false
	}

	// Check collision with snake bodies (skip tails that will move)
	if state.Board.IsOccupied(nextPos, true) {
		return false
	}

	return true
}

// GetValidMoves returns all valid moves for a snake
func GetValidMoves(state *board.GameState, snakeID string) []string {
	validMoves := make([]string, 0, 4)

	for _, move := range board.AllMoves() {
		if IsMoveValid(state, snakeID, move) {
			validMoves = append(validMoves, move)
		}
	}

	return validMoves
}

// SimulateMultipleMoves simulates moves for all snakes simultaneously
// This is useful for lookahead in multi-agent scenarios
func SimulateMultipleMoves(state *board.GameState, moves map[string]string) *board.GameState {
	// For simplicity, apply moves sequentially
	// In a more sophisticated implementation, we'd apply all moves truly simultaneously
	newState := state

	for snakeID, move := range moves {
		newState = SimulateMove(newState, snakeID, move)
	}

	return newState
}
