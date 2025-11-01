package main

import (
	"log"
	"time"
	
	"github.com/ErwinsExpertise/go-battleclank/algorithms/search"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/ErwinsExpertise/go-battleclank/telemetry"
)

// moveRefactored is the new modular decision-making function
// This replaces the monolithic logic.go implementation with a clean, modular approach
func moveRefactored(state GameState) BattlesnakeMoveResponse {
	startTime := time.Now()
	
	// Convert API state to internal board representation
	internalState := convertToInternalState(state)
	
	// Use greedy search with tuned penalties
	strategy := search.NewGreedySearch()
	
	// Find best move
	bestMove := strategy.FindBestMove(internalState)
	
	// Calculate execution time
	executionTime := time.Since(startTime)
	
	// Log decision (telemetry)
	logger := telemetry.NewLogger(true, false)
	
	// Calculate all move scores for telemetry
	allScores := make(map[string]float64)
	for _, move := range board.AllMoves() {
		allScores[move] = strategy.ScoreMove(internalState, move)
	}
	
	decision := telemetry.MoveDecision{
		Timestamp:        startTime,
		GameID:           state.Game.ID,
		Turn:             state.Turn,
		ChosenMove:       bestMove,
		ChosenScore:      allScores[bestMove],
		AlternativeMoves: allScores,
		ExecutionTime:    executionTime,
	}
	
	logger.LogMoveDecision(decision)
	
	return BattlesnakeMoveResponse{
		Move: bestMove,
	}
}

// convertToInternalState converts API GameState to internal board.GameState
func convertToInternalState(apiState GameState) *board.GameState {
	// Convert snakes
	snakes := make([]board.Snake, len(apiState.Board.Snakes))
	for i, apiSnake := range apiState.Board.Snakes {
		body := make([]board.Coord, len(apiSnake.Body))
		for j, coord := range apiSnake.Body {
			body[j] = board.Coord{X: coord.X, Y: coord.Y}
		}
		
		snakes[i] = board.Snake{
			ID:     apiSnake.ID,
			Name:   apiSnake.Name,
			Health: apiSnake.Health,
			Body:   body,
			Head:   board.Coord{X: apiSnake.Head.X, Y: apiSnake.Head.Y},
			Length: apiSnake.Length,
		}
	}
	
	// Convert food
	food := make([]board.Coord, len(apiState.Board.Food))
	for i, f := range apiState.Board.Food {
		food[i] = board.Coord{X: f.X, Y: f.Y}
	}
	
	// Convert hazards
	hazards := make([]board.Coord, len(apiState.Board.Hazards))
	for i, h := range apiState.Board.Hazards {
		hazards[i] = board.Coord{X: h.X, Y: h.Y}
	}
	
	// Convert You snake
	youBody := make([]board.Coord, len(apiState.You.Body))
	for i, coord := range apiState.You.Body {
		youBody[i] = board.Coord{X: coord.X, Y: coord.Y}
	}
	
	you := board.Snake{
		ID:     apiState.You.ID,
		Name:   apiState.You.Name,
		Health: apiState.You.Health,
		Body:   youBody,
		Head:   board.Coord{X: apiState.You.Head.X, Y: apiState.You.Head.Y},
		Length: apiState.You.Length,
	}
	
	return &board.GameState{
		Turn: apiState.Turn,
		Board: board.Board{
			Height:  apiState.Board.Height,
			Width:   apiState.Board.Width,
			Food:    food,
			Hazards: hazards,
			Snakes:  snakes,
		},
		You: you,
	}
}

// startRefactored is called when game starts (stateless)
func startRefactored(state GameState) {
	log.Printf("GAME START: %s (Refactored Engine)", state.Game.ID)
}

// endRefactored is called when game ends (stateless)
func endRefactored(state GameState) {
	log.Printf("GAME OVER: %s (Refactored Engine)", state.Game.ID)
}
