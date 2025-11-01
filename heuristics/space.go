package heuristics

import "github.com/ErwinsExpertise/go-battleclank/engine/board"

// Package heuristics provides reusable heuristic functions for evaluating game states

// FloodFill counts reachable spaces from a position using iterative BFS
// This matches the baseline's algorithm (VecDeque-based BFS, not recursion)
// maxDepth limits the search for performance (baseline caps at 121 for 11x11 board)
func FloodFill(state *board.GameState, start board.Coord, maxDepth int) int {
	// Check if starting position is valid and not blocked
	if !state.Board.IsInBounds(start) || state.Board.IsOccupied(start, true) {
		return 0
	}
	
	// Use slice as queue (more efficient than recursive approach)
	queue := []board.Coord{start}
	visited := make(map[board.Coord]bool)
	visited[start] = true
	
	count := 0
	
	// Iterative BFS matching baseline implementation
	for len(queue) > 0 {
		// Pop front (FIFO for BFS)
		pos := queue[0]
		queue = queue[1:]
		count++
		
		// Cap at maxDepth for performance (baseline uses 121)
		if count >= maxDepth {
			break
		}
		
		// Check all 4 neighbors
		for _, neighbor := range state.Board.GetNeighbors(pos) {
			// Skip if already visited
			if visited[neighbor] {
				continue
			}
			
			// Check if position is valid
			if !state.Board.IsInBounds(neighbor) {
				continue
			}
			
			// Check if blocked by snake (skip tails that will move)
			if state.Board.IsOccupied(neighbor, true) {
				continue
			}
			
			// Add to queue and mark visited
			queue = append(queue, neighbor)
			visited[neighbor] = true
		}
	}
	
	return count
}

// FloodFillForSnake calculates reachable space for a specific snake using iterative BFS
// This is useful for trap detection and opponent space analysis
func FloodFillForSnake(state *board.GameState, snakeID string, start board.Coord, maxDepth int) int {
	queue := []board.Coord{start}
	visited := make(map[board.Coord]bool)
	visited[start] = true
	
	count := 0
	
	// Iterative BFS
	for len(queue) > 0 {
		pos := queue[0]
		queue = queue[1:]
		count++
		
		if count >= maxDepth {
			break
		}
		
		for _, neighbor := range state.Board.GetNeighbors(pos) {
			if visited[neighbor] || !state.Board.IsInBounds(neighbor) {
				continue
			}
			
			// Check if blocked by snake bodies (skip the specific snake's tail)
			isBlocked := false
			for _, snake := range state.Board.Snakes {
				for i, segment := range snake.Body {
					// Skip the tail of the snake we're evaluating for
					if snake.ID == snakeID && i == len(snake.Body)-1 {
						continue
					}
					// Skip other snakes' tails that will move
					if i == len(snake.Body)-1 && snake.Health != 100 {
						continue
					}
					if neighbor.X == segment.X && neighbor.Y == segment.Y {
						isBlocked = true
						break
					}
				}
				if isBlocked {
					break
				}
			}
			
			if !isBlocked {
				queue = append(queue, neighbor)
				visited[neighbor] = true
			}
		}
	}
	
	return count
}

// EvaluateSpace returns a normalized score (0-1) for available space
func EvaluateSpace(state *board.GameState, pos board.Coord, maxDepth int) float64 {
	reachable := FloodFill(state, pos, maxDepth)
	totalSpaces := state.Board.Width * state.Board.Height
	return float64(reachable) / float64(totalSpaces)
}

// CompareSpace compares reachable space between two positions
// Returns the difference (pos1Space - pos2Space)
func CompareSpace(state *board.GameState, pos1, pos2 board.Coord, maxDepth int) int {
	space1 := FloodFill(state, pos1, maxDepth)
	space2 := FloodFill(state, pos2, maxDepth)
	return space1 - space2
}

// SpaceTrapLevel represents trap danger levels based on space ratios
type SpaceTrapLevel int

const (
	SpaceSafe SpaceTrapLevel = iota        // 80%+ ratio - safe
	SpaceModerate                           // 60-80% ratio - moderate concern
	SpaceSevere                             // 40-60% ratio - severe danger
	SpaceCritical                           // <40% ratio - critical trap
)

// SpaceTrapPenalties maps trap levels to penalty values
// MATCHED TO BASELINE: trap_penalty=-250, severe=-450, critical=-600
var SpaceTrapPenalties = map[SpaceTrapLevel]float64{
	SpaceSafe:     0.0,    // 80%+ ratio - safe
	SpaceModerate: 250.0,  // 60-80% ratio - baseline: -250
	SpaceSevere:   450.0,  // 40-60% ratio - baseline: -450
	SpaceCritical: 600.0,  // <40% ratio - baseline: -600
}

// EvaluateSpaceRatio calculates space-to-body-length ratio and returns trap level
// This matches the baseline snake's ratio-based trap detection
func EvaluateSpaceRatio(state *board.GameState, pos board.Coord, maxDepth int) (float64, SpaceTrapLevel) {
	reachableSpace := FloodFill(state, pos, maxDepth)
	bodyLength := state.You.Length
	
	// Calculate ratio (tail moves, so don't need 100% of body length)
	ratio := float64(reachableSpace) / float64(bodyLength)
	
	// Determine trap level based on ratio thresholds
	var trapLevel SpaceTrapLevel
	if ratio < 0.4 {
		trapLevel = SpaceCritical // Less than 40% - critical danger
	} else if ratio < 0.6 {
		trapLevel = SpaceSevere // 40-60% - severe danger
	} else if ratio < 0.8 {
		trapLevel = SpaceModerate // 60-80% - moderate concern
	} else {
		trapLevel = SpaceSafe // 80%+ - good space
	}
	
	return ratio, trapLevel
}

// GetSpaceTrapPenalty returns the penalty for a given trap level
func GetSpaceTrapPenalty(trapLevel SpaceTrapLevel) float64 {
	return SpaceTrapPenalties[trapLevel]
}

// EvaluateFoodTrapRatio evaluates if eating food would trap the snake
// Uses 70% threshold since tail doesn't move after eating
func EvaluateFoodTrapRatio(state *board.GameState, pos board.Coord, maxDepth int) (bool, float64) {
	// Simulate space after eating (tail stays in place)
	spacesAfterEating := FloodFillAfterEating(state, pos, maxDepth)
	bodyLength := state.You.Length
	
	// Need 70% of body length since tail doesn't move
	minRequired := int(float64(bodyLength) * 0.7)
	
	isTrap := spacesAfterEating < minRequired
	ratio := float64(spacesAfterEating) / float64(bodyLength)
	
	return isTrap, ratio
}

// FloodFillAfterEating simulates flood fill after eating food (tail doesn't move)
// Uses iterative BFS to match baseline approach
func FloodFillAfterEating(state *board.GameState, pos board.Coord, maxDepth int) int {
	// Mark all snake bodies as blocked (including tails, since they won't move after eating)
	blocked := make(map[board.Coord]bool)
	for _, snake := range state.Board.Snakes {
		for _, segment := range snake.Body {
			blocked[segment] = true
		}
	}
	
	// Start position should not be blocked
	delete(blocked, pos)
	
	// Iterative BFS
	queue := []board.Coord{pos}
	visited := make(map[board.Coord]bool)
	visited[pos] = true
	
	count := 0
	
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		count++
		
		if count >= maxDepth {
			break
		}
		
		for _, neighbor := range state.Board.GetNeighbors(current) {
			if visited[neighbor] || !state.Board.IsInBounds(neighbor) || blocked[neighbor] {
				continue
			}
			
			queue = append(queue, neighbor)
			visited[neighbor] = true
		}
	}
	
	return count
}
