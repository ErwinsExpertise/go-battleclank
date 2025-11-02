//go:build cuda
// +build cuda

package gpu

import (
	"fmt"
	"unsafe"
	
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/mumax/3/cuda"
)

// BoardStateGPU represents a board state optimized for GPU processing
type BoardStateGPU struct {
	Width        int
	Height       int
	Occupancy    []float32       // CPU copy of occupancy grid
	NumSnakes    int
	OccupancyGPU unsafe.Pointer  // GPU memory pointer for occupancy grid
	usingGPU     bool            // Track if GPU memory is allocated
}

// NewBoardStateGPU converts CPU board state to GPU-optimized format
// Allocates GPU memory and uploads data when GPU is available
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
	
	boardState := &BoardStateGPU{
		Width:     w,
		Height:    h,
		Occupancy: occupancy,
		NumSnakes: len(state.Board.Snakes),
		usingGPU:  false,
	}
	
	// Allocate GPU memory and upload data
	if GPUAvailable {
		defer func() {
			if r := recover(); r != nil {
				// If GPU allocation fails, continue with CPU-only mode
				boardState.usingGPU = false
			}
		}()
		
		// Allocate GPU memory for occupancy grid
		bytes := int64(size * 4) // 4 bytes per float32
		occupancyGPU := cuda.MemAlloc(bytes)
		
		// Copy data to GPU
		cuda.Memcpy(occupancyGPU, unsafe.Pointer(&occupancy[0]), bytes)
		
		boardState.OccupancyGPU = occupancyGPU
		boardState.usingGPU = true
	}
	
	return boardState, nil
}

// Free releases GPU memory
func (b *BoardStateGPU) Free() {
	if b.usingGPU && b.OccupancyGPU != nil {
		cuda.MemFree(b.OccupancyGPU)
		b.OccupancyGPU = nil
		b.usingGPU = false
	}
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
