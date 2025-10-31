package heuristics

import "github.com/ErwinsExpertise/go-battleclank/engine/board"

// Package heuristics provides reusable heuristic functions for evaluating game states

// FloodFill counts reachable spaces from a position using BFS/flood-fill algorithm
// maxDepth limits the search depth for performance
func FloodFill(state *board.GameState, start board.Coord, maxDepth int) int {
	visited := make(map[board.Coord]bool)
	return floodFillRecursive(state, start, visited, 0, maxDepth)
}

// floodFillRecursive is the recursive helper for FloodFill
func floodFillRecursive(state *board.GameState, pos board.Coord, visited map[board.Coord]bool, depth int, maxDepth int) int {
	// Limit recursion depth
	if depth > maxDepth {
		return 0
	}
	
	// Check if position is valid and unvisited
	if !state.Board.IsInBounds(pos) || visited[pos] {
		return 0
	}
	
	// Check if blocked by snake (skip tails)
	if state.Board.IsOccupied(pos, true) {
		return 0
	}
	
	visited[pos] = true
	count := 1
	
	// Recursively check neighbors
	for _, neighbor := range state.Board.GetNeighbors(pos) {
		count += floodFillRecursive(state, neighbor, visited, depth+1, maxDepth)
	}
	
	return count
}

// FloodFillForSnake calculates reachable space for a specific snake
// This is useful for trap detection and opponent space analysis
func FloodFillForSnake(state *board.GameState, snakeID string, start board.Coord, maxDepth int) int {
	visited := make(map[board.Coord]bool)
	return floodFillForSnakeRecursive(state, snakeID, start, visited, 0, maxDepth)
}

func floodFillForSnakeRecursive(state *board.GameState, snakeID string, pos board.Coord, visited map[board.Coord]bool, depth int, maxDepth int) int {
	if depth > maxDepth {
		return 0
	}
	
	if !state.Board.IsInBounds(pos) || visited[pos] {
		return 0
	}
	
	// Check if blocked by snake bodies (skip the specific snake's tail)
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
			if pos.X == segment.X && pos.Y == segment.Y {
				return 0
			}
		}
	}
	
	visited[pos] = true
	count := 1
	
	for _, neighbor := range state.Board.GetNeighbors(pos) {
		count += floodFillForSnakeRecursive(state, snakeID, neighbor, visited, depth+1, maxDepth)
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
