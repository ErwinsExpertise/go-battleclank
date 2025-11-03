//go:build !cuda
// +build !cuda

package gpu

import (
	"fmt"
	
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

// BoardStateGPU represents a board state optimized for GPU processing
// Non-CUDA build - stores data in CPU memory only
type BoardStateGPU struct {
	Width     int
	Height    int
	Occupancy []float32 // Flattened 2D grid: 0=empty, 1=occupied
	NumSnakes int
}

// NewBoardStateGPU converts CPU board state to GPU-optimized format
// Non-CUDA build - stores data in CPU memory only
func NewBoardStateGPU(state *board.GameState) (*BoardStateGPU, error) {
	if !IsAvailable() {
		return nil, fmt.Errorf("GPU not available")
	}
	
	w, h := state.Board.Width, state.Board.Height
	size := w * h
	
	// Create occupancy grid (flattened 2D array)
	occupancy := make([]float32, size)
	
	// Mark all snake body segments as occupied
	for _, snake := range state.Board.Snakes {
		for _, segment := range snake.Body {
			idx := segment.Y*w + segment.X
			if idx >= 0 && idx < size {
				occupancy[idx] = 1.0
			}
		}
	}
	
	return &BoardStateGPU{
		Width:     w,
		Height:    h,
		Occupancy: occupancy,
		NumSnakes: len(state.Board.Snakes),
	}, nil
}

// Free releases GPU memory
// Non-CUDA build - no-op since no GPU memory is allocated
func (b *BoardStateGPU) Free() {
	// No-op in non-CUDA build
}

// IsOccupied checks if a position is occupied on the board
func (b *BoardStateGPU) IsOccupied(x, y int) bool {
	if x < 0 || x >= b.Width || y < 0 || y >= b.Height {
		return true // Out of bounds counts as occupied
	}
	idx := y*b.Width + x
	return b.Occupancy[idx] > 0
}

// Clone creates a copy of the board state
func (b *BoardStateGPU) Clone() *BoardStateGPU {
	occupancyCopy := make([]float32, len(b.Occupancy))
	copy(occupancyCopy, b.Occupancy)
	
	return &BoardStateGPU{
		Width:     b.Width,
		Height:    b.Height,
		Occupancy: occupancyCopy,
		NumSnakes: b.NumSnakes,
	}
}
