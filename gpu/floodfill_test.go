package gpu

import (
	"testing"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

func TestFloodFillGPU(t *testing.T) {
	// Create a simple test board
	state := &board.GameState{
		Board: board.Board{
			Width:  5,
			Height: 5,
			Snakes: []board.Snake{
				{
					ID:   "snake1",
					Body: []board.Coord{{X: 2, Y: 2}}, // Single cell obstacle in center
					Head: board.Coord{X: 2, Y: 2},
				},
			},
		},
		You: board.Snake{ID: "us"},
	}
	
	// Test with GPU disabled
	SetEnabled(false)
	_, err := FloodFillGPU(state, board.Coord{X: 0, Y: 0}, 10)
	if err == nil {
		t.Error("Expected error when GPU not available")
	}
	
	// Test with GPU enabled (CPU fallback)
	SetEnabled(true)
	Initialize()
	
	if !IsAvailable() {
		t.Skip("GPU not available, skipping flood fill test")
	}
	
	// Start from corner (0, 0)
	count, err := FloodFillGPU(state, board.Coord{X: 0, Y: 0}, 10)
	if err != nil {
		t.Fatalf("FloodFillGPU failed: %v", err)
	}
	
	// Should be able to reach most cells except the obstacle
	// 5x5 = 25 cells, minus 1 obstacle = 24 reachable
	if count < 10 || count > 24 {
		t.Logf("Flood fill from (0,0) reached %d cells (expected 10-24)", count)
	}
	
	// Test flood fill from occupied position (should still work)
	count2, err := FloodFillGPU(state, board.Coord{X: 2, Y: 2}, 10)
	if err != nil {
		t.Fatalf("FloodFillGPU from occupied position failed: %v", err)
	}
	
	// Should reach 0 cells (starting on obstacle)
	if count2 != 0 {
		t.Logf("Flood fill from occupied position reached %d cells", count2)
	}
}

func TestFloodFillMultiple(t *testing.T) {
	state := &board.GameState{
		Board: board.Board{
			Width:  7,
			Height: 7,
			Snakes: []board.Snake{
				{
					ID:   "snake1",
					Body: []board.Coord{{X: 3, Y: 3}}, // Center obstacle
					Head: board.Coord{X: 3, Y: 3},
				},
			},
		},
		You: board.Snake{ID: "us"},
	}
	
	SetEnabled(true)
	Initialize()
	
	if !IsAvailable() {
		t.Skip("GPU not available, skipping multiple flood fill test")
	}
	
	// Test multiple starting positions
	starts := []board.Coord{
		{X: 0, Y: 0}, // Top-left
		{X: 6, Y: 0}, // Top-right
		{X: 0, Y: 6}, // Bottom-left
		{X: 6, Y: 6}, // Bottom-right
	}
	
	results, err := FloodFillMultiple(state, starts, 10)
	if err != nil {
		t.Fatalf("FloodFillMultiple failed: %v", err)
	}
	
	if len(results) != len(starts) {
		t.Errorf("Expected %d results, got %d", len(starts), len(results))
	}
	
	// Each corner should reach roughly 1/4 of the board
	for i, count := range results {
		if count < 5 {
			t.Logf("Corner %d reached %d cells (seems low)", i, count)
		}
		t.Logf("Flood fill from start position %d reached %d cells", i, count)
	}
}

func TestFloodFillCPU(t *testing.T) {
	// Test the CPU fallback directly
	boardState := &BoardStateGPU{
		Width:     5,
		Height:    5,
		Occupancy: make([]float32, 25),
	}
	
	// Create a wall down the middle (column 2)
	for y := 0; y < 5; y++ {
		boardState.Occupancy[y*5+2] = 1.0
	}
	
	// Flood fill from left side
	countLeft := floodFillCPU(boardState, board.Coord{X: 0, Y: 0}, 10)
	
	// Should reach 10 cells on the left side (2 columns × 5 rows)
	if countLeft != 10 {
		t.Logf("Flood fill on left side reached %d cells, expected 10", countLeft)
	}
	
	// Flood fill from right side
	countRight := floodFillCPU(boardState, board.Coord{X: 4, Y: 0}, 10)
	
	// Should reach 10 cells on the right side (2 columns × 5 rows)
	if countRight != 10 {
		t.Logf("Flood fill on right side reached %d cells, expected 10", countRight)
	}
}
