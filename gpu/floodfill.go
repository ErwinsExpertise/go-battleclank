package gpu

import (
	"fmt"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

// FloodFillGPU performs parallel flood fill on GPU (or CPU fallback)
// Returns the number of reachable cells from the starting position
func FloodFillGPU(state *board.GameState, start board.Coord, maxDepth int) (int, error) {
	if !IsAvailable() {
		return 0, fmt.Errorf("GPU not available")
	}
	
	// Convert board to GPU format
	boardGPU, err := NewBoardStateGPU(state)
	if err != nil {
		return 0, err
	}
	defer boardGPU.Free()
	
	// Use CPU-based BFS for now
	// Future: This will use GPU kernels for parallel processing
	return floodFillCPU(boardGPU, start, maxDepth), nil
}

// floodFillCPU performs CPU-based flood fill (fallback implementation)
func floodFillCPU(boardState *BoardStateGPU, start board.Coord, maxDepth int) int {
	visited := make(map[int]bool)
	frontier := []board.Coord{start}
	
	depth := 0
	for depth < maxDepth && len(frontier) > 0 {
		nextFrontier := make([]board.Coord, 0)
		
		for _, pos := range frontier {
			idx := pos.Y*boardState.Width + pos.X
			
			// Skip if already visited or out of bounds
			if visited[idx] {
				continue
			}
			if pos.X < 0 || pos.X >= boardState.Width || pos.Y < 0 || pos.Y >= boardState.Height {
				continue
			}
			
			// Skip if occupied
			if boardState.IsOccupied(pos.X, pos.Y) {
				continue
			}
			
			// Mark as visited
			visited[idx] = true
			
			// Add neighbors to next frontier
			neighbors := []board.Coord{
				{X: pos.X, Y: pos.Y + 1},     // up
				{X: pos.X, Y: pos.Y - 1},     // down
				{X: pos.X - 1, Y: pos.Y},     // left
				{X: pos.X + 1, Y: pos.Y},     // right
			}
			
			for _, neighbor := range neighbors {
				// Check bounds before calculating index
				if neighbor.X < 0 || neighbor.X >= boardState.Width || neighbor.Y < 0 || neighbor.Y >= boardState.Height {
					continue
				}
				neighborIdx := neighbor.Y*boardState.Width + neighbor.X
				if !visited[neighborIdx] && boardState.Occupancy[neighborIdx] == 0 {
					nextFrontier = append(nextFrontier, neighbor)
				}
			}
		}
		
		frontier = nextFrontier
		depth++
	}
	
	return len(visited)
}

// FloodFillMultiple performs flood fill from multiple starting positions
// This is useful for comparing space availability for different moves
func FloodFillMultiple(state *board.GameState, starts []board.Coord, maxDepth int) ([]int, error) {
	if !IsAvailable() {
		return nil, fmt.Errorf("GPU not available")
	}
	
	results := make([]int, len(starts))
	
	// For now, process each start position sequentially
	// Future: Process all positions in parallel on GPU
	for i, start := range starts {
		count, err := FloodFillGPU(state, start, maxDepth)
		if err != nil {
			return nil, err
		}
		results[i] = count
	}
	
	return results, nil
}
