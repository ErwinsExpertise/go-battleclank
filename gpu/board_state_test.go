package gpu

import (
	"testing"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

func TestNewBoardStateGPU(t *testing.T) {
	// Setup test state
	state := &board.GameState{
		Board: board.Board{
			Width:  11,
			Height: 11,
			Snakes: []board.Snake{
				{
					ID:   "snake1",
					Body: []board.Coord{{X: 5, Y: 5}, {X: 5, Y: 4}, {X: 5, Y: 3}},
					Head: board.Coord{X: 5, Y: 5},
				},
			},
		},
	}
	
	// Test with GPU disabled
	SetEnabled(false)
	_, err := NewBoardStateGPU(state)
	if err == nil {
		t.Error("Expected error when GPU not available")
	}
	
	// Test with GPU enabled (will use CPU fallback)
	SetEnabled(true)
	Initialize()
	
	if !IsAvailable() {
		t.Skip("GPU not available, skipping GPU board state test")
	}
	
	boardGPU, err := NewBoardStateGPU(state)
	if err != nil {
		t.Fatalf("Failed to create board state: %v", err)
	}
	defer boardGPU.Free()
	
	// Verify dimensions
	if boardGPU.Width != 11 || boardGPU.Height != 11 {
		t.Errorf("Expected 11x11 board, got %dx%d", boardGPU.Width, boardGPU.Height)
	}
	
	// Verify occupancy grid
	if len(boardGPU.Occupancy) != 121 {
		t.Errorf("Expected 121 cells, got %d", len(boardGPU.Occupancy))
	}
	
	// Check that snake positions are marked as occupied
	for _, segment := range state.Board.Snakes[0].Body {
		if !boardGPU.IsOccupied(segment.X, segment.Y) {
			t.Errorf("Position (%d, %d) should be occupied", segment.X, segment.Y)
		}
	}
	
	// Check that empty positions are not occupied
	if boardGPU.IsOccupied(0, 0) {
		t.Error("Position (0, 0) should not be occupied")
	}
}

func TestBoardStateGPU_IsOccupied(t *testing.T) {
	boardState := &BoardStateGPU{
		Width:     5,
		Height:    5,
		Occupancy: make([]float32, 25),
	}
	
	// Mark position (2, 2) as occupied
	boardState.Occupancy[2*5+2] = 1.0
	
	tests := []struct {
		name     string
		x, y     int
		expected bool
	}{
		{"Occupied cell", 2, 2, true},
		{"Empty cell", 1, 1, false},
		{"Out of bounds negative", -1, 0, true},
		{"Out of bounds positive X", 5, 0, true},
		{"Out of bounds positive Y", 0, 5, true},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := boardState.IsOccupied(tt.x, tt.y)
			if result != tt.expected {
				t.Errorf("IsOccupied(%d, %d) = %v, expected %v", tt.x, tt.y, result, tt.expected)
			}
		})
	}
}

func TestBoardStateGPU_Clone(t *testing.T) {
	original := &BoardStateGPU{
		Width:     3,
		Height:    3,
		Occupancy: []float32{1, 0, 0, 0, 1, 0, 0, 0, 1},
		NumSnakes: 2,
	}
	
	clone := original.Clone()
	
	// Verify clone has same values
	if clone.Width != original.Width || clone.Height != original.Height {
		t.Error("Clone has different dimensions")
	}
	
	if len(clone.Occupancy) != len(original.Occupancy) {
		t.Error("Clone has different occupancy length")
	}
	
	for i := range original.Occupancy {
		if clone.Occupancy[i] != original.Occupancy[i] {
			t.Errorf("Clone occupancy differs at index %d", i)
		}
	}
	
	// Verify clone is independent (modifying clone doesn't affect original)
	clone.Occupancy[0] = 99.0
	if original.Occupancy[0] == 99.0 {
		t.Error("Modifying clone affected original")
	}
}
