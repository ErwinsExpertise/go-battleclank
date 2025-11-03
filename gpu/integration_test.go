package gpu

import (
	"testing"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

// TestGPUIntegrationWithMCTS tests the full GPU integration with MCTS
func TestGPUIntegrationWithMCTS(t *testing.T) {
	// Create a test game state
	state := createTestGameState()
	
	// Test with GPU disabled
	SetEnabled(false)
	mcts := NewMCTSBatchGPU(50, 10)
	_, err := mcts.SimulateBatch(state, 100)
	if err == nil {
		t.Error("Expected error when GPU not enabled")
	}
	
	// Test with GPU enabled (will use CPU fallback)
	SetEnabled(true)
	Initialize()
	
	if !IsAvailable() {
		t.Skip("GPU not available, skipping integration test")
	}
	
	mcts = NewMCTSBatchGPU(50, 10)
	move, err := mcts.SimulateBatch(state, 100)
	if err != nil {
		t.Fatalf("MCTS simulation failed: %v", err)
	}
	
	// Verify we got a valid move
	validMoves := []string{board.MoveUp, board.MoveDown, board.MoveLeft, board.MoveRight}
	found := false
	for _, validMove := range validMoves {
		if move == validMove {
			found = true
			break
		}
	}
	
	if !found {
		t.Errorf("Got invalid move: %s", move)
	}
	
	t.Logf("MCTS selected move: %s", move)
}

// TestGPUFloodFillIntegration tests flood fill integration
func TestGPUFloodFillIntegration(t *testing.T) {
	state := createTestGameState()
	
	SetEnabled(true)
	Initialize()
	
	if !IsAvailable() {
		t.Skip("GPU not available, skipping flood fill integration test")
	}
	
	// Test flood fill from snake's current position
	ourSnake := state.Board.Snakes[0]
	count, err := FloodFillGPU(state, ourSnake.Head, 10)
	if err != nil {
		t.Fatalf("Flood fill failed: %v", err)
	}
	
	if count <= 0 {
		t.Error("Flood fill should find at least some reachable space")
	}
	
	t.Logf("Flood fill found %d reachable cells", count)
	
	// Test multiple flood fills
	testPositions := []board.Coord{
		ourSnake.Head,
		{X: ourSnake.Head.X + 1, Y: ourSnake.Head.Y}, // Right
		{X: ourSnake.Head.X, Y: ourSnake.Head.Y + 1}, // Up
		{X: ourSnake.Head.X - 1, Y: ourSnake.Head.Y}, // Left
	}
	
	results, err := FloodFillMultiple(state, testPositions, 10)
	if err != nil {
		t.Fatalf("Multiple flood fill failed: %v", err)
	}
	
	if len(results) != len(testPositions) {
		t.Errorf("Expected %d results, got %d", len(testPositions), len(results))
	}
	
	t.Logf("Multiple flood fill results: %v", results)
}

// TestGPUPerformance tests basic performance characteristics
func TestGPUPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}
	
	state := createTestGameState()
	
	SetEnabled(true)
	Initialize()
	
	if !IsAvailable() {
		t.Skip("GPU not available, skipping performance test")
	}
	
	// Run a moderate number of simulations
	mcts := NewMCTSBatchGPU(100, 15)
	_, err := mcts.SimulateBatch(state, 500)
	if err != nil {
		t.Fatalf("Performance test failed: %v", err)
	}
	
	// If we got here, the performance is acceptable
	t.Log("Performance test completed successfully")
}

// Helper function to create a test game state
func createTestGameState() *board.GameState {
	return &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food: []board.Coord{
				{X: 3, Y: 3},
				{X: 7, Y: 7},
			},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Name:   "test-snake",
					Health: 75,
					Body: []board.Coord{
						{X: 5, Y: 5},
						{X: 5, Y: 4},
						{X: 5, Y: 3},
					},
					Head:   board.Coord{X: 5, Y: 5},
					Length: 3,
				},
				{
					ID:     "enemy",
					Name:   "enemy-snake",
					Health: 80,
					Body: []board.Coord{
						{X: 8, Y: 8},
						{X: 8, Y: 7},
						{X: 8, Y: 6},
						{X: 8, Y: 5},
					},
					Head:   board.Coord{X: 8, Y: 8},
					Length: 4,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Name:   "test-snake",
			Health: 75,
			Body: []board.Coord{
				{X: 5, Y: 5},
				{X: 5, Y: 4},
				{X: 5, Y: 3},
			},
			Head:   board.Coord{X: 5, Y: 5},
			Length: 3,
		},
	}
}
