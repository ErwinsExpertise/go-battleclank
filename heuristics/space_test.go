package heuristics

import (
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"testing"
)

func TestFloodFill(t *testing.T) {
	// Create a simple board with a snake
	state := &board.GameState{
		Board: board.Board{
			Width:  11,
			Height: 11,
			Snakes: []board.Snake{
				{
					ID:     "snake1",
					Health: 90,
					Body:   []board.Coord{{X: 5, Y: 5}, {X: 5, Y: 4}, {X: 5, Y: 3}},
					Head:   board.Coord{X: 5, Y: 5},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "you",
			Health: 100,
			Body:   []board.Coord{{X: 0, Y: 0}, {X: 0, Y: 1}, {X: 0, Y: 2}},
			Head:   board.Coord{X: 0, Y: 0},
			Length: 3,
		},
	}
	
	// Test flood fill from an open position
	count := FloodFill(state, board.Coord{X: 0, Y: 0}, 10)
	
	if count <= 0 {
		t.Errorf("FloodFill returned %d, expected positive count", count)
	}
	
	// Test flood fill from blocked position
	count2 := FloodFill(state, board.Coord{X: 5, Y: 5}, 10)
	if count2 != 0 {
		t.Errorf("FloodFill from blocked position returned %d, expected 0", count2)
	}
}

func TestEvaluateSpace(t *testing.T) {
	state := &board.GameState{
		Board: board.Board{
			Width:  11,
			Height: 11,
			Snakes: []board.Snake{},
		},
		You: board.Snake{
			ID:     "you",
			Head:   board.Coord{X: 5, Y: 5},
			Length: 3,
		},
	}
	
	// In an empty board, should have access to most of the space
	score := EvaluateSpace(state, board.Coord{X: 5, Y: 5}, 10)
	
	if score <= 0 || score > 1 {
		t.Errorf("EvaluateSpace returned %.2f, expected value between 0 and 1", score)
	}
	
	// Should have significant space in open board
	if score < 0.5 {
		t.Errorf("EvaluateSpace returned %.2f, expected > 0.5 in open board", score)
	}
}

func TestCompareSpace(t *testing.T) {
	state := &board.GameState{
		Board: board.Board{
			Width:  11,
			Height: 11,
			Snakes: []board.Snake{
				{
					ID:     "snake1",
					Health: 90,
					Body:   []board.Coord{{X: 10, Y: 10}, {X: 10, Y: 9}, {X: 10, Y: 8}},
					Head:   board.Coord{X: 10, Y: 10},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "you",
			Health: 100,
			Head:   board.Coord{X: 0, Y: 0},
			Length: 3,
		},
	}
	
	// Position near corner vs center should have different space
	pos1 := board.Coord{X: 0, Y: 0}
	pos2 := board.Coord{X: 5, Y: 5}
	
	diff := CompareSpace(state, pos1, pos2, 10)
	
	// Center should have more space than corner
	if diff > 0 {
		t.Errorf("CompareSpace returned %d, expected negative (center has more space)", diff)
	}
}

func TestFloodFillForSnake(t *testing.T) {
	state := &board.GameState{
		Board: board.Board{
			Width:  11,
			Height: 11,
			Snakes: []board.Snake{
				{
					ID:     "snake1",
					Health: 90,
					Body:   []board.Coord{{X: 5, Y: 5}, {X: 5, Y: 4}, {X: 5, Y: 3}},
					Head:   board.Coord{X: 5, Y: 5},
					Length: 3,
				},
				{
					ID:     "you",
					Health: 90,
					Body:   []board.Coord{{X: 0, Y: 0}, {X: 0, Y: 1}, {X: 0, Y: 2}},
					Head:   board.Coord{X: 0, Y: 0},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "you",
			Health: 90,
			Body:   []board.Coord{{X: 0, Y: 0}, {X: 0, Y: 1}, {X: 0, Y: 2}},
			Head:   board.Coord{X: 0, Y: 0},
			Length: 3,
		},
	}
	
	// Test flood fill for our snake from a valid adjacent position
	// Starting from (1, 0) which is adjacent to head at (0, 0)
	count := FloodFillForSnake(state, "you", board.Coord{X: 1, Y: 0}, 10)
	
	if count <= 0 {
		t.Errorf("FloodFillForSnake returned %d, expected positive count", count)
	}
	
	// Test for enemy snake from an adjacent position
	// Starting from (6, 5) which is adjacent to enemy head at (5, 5)
	enemyCount := FloodFillForSnake(state, "snake1", board.Coord{X: 6, Y: 5}, 10)
	
	if enemyCount <= 0 {
		t.Errorf("FloodFillForSnake for enemy returned %d, expected positive count", enemyCount)
	}
}
