package main

import (
	"testing"
)

// Test info function
func TestInfo(t *testing.T) {
	response := info()
	
	if response.APIVersion != "1" {
		t.Errorf("Expected APIVersion to be '1', got '%s'", response.APIVersion)
	}
	
	if response.Color == "" {
		t.Error("Color should not be empty")
	}
	
	if response.Head == "" {
		t.Error("Head should not be empty")
	}
	
	if response.Tail == "" {
		t.Error("Tail should not be empty")
	}
}

// Test getNextPosition
func TestGetNextPosition(t *testing.T) {
	tests := []struct {
		name     string
		pos      Coord
		move     string
		expected Coord
	}{
		{"Move up", Coord{X: 5, Y: 5}, MoveUp, Coord{X: 5, Y: 6}},
		{"Move down", Coord{X: 5, Y: 5}, MoveDown, Coord{X: 5, Y: 4}},
		{"Move left", Coord{X: 5, Y: 5}, MoveLeft, Coord{X: 4, Y: 5}},
		{"Move right", Coord{X: 5, Y: 5}, MoveRight, Coord{X: 6, Y: 5}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getNextPosition(tt.pos, tt.move)
			if result.X != tt.expected.X || result.Y != tt.expected.Y {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Test manhattanDistance
func TestManhattanDistance(t *testing.T) {
	tests := []struct {
		name     string
		a        Coord
		b        Coord
		expected int
	}{
		{"Same position", Coord{X: 0, Y: 0}, Coord{X: 0, Y: 0}, 0},
		{"Horizontal distance", Coord{X: 0, Y: 0}, Coord{X: 5, Y: 0}, 5},
		{"Vertical distance", Coord{X: 0, Y: 0}, Coord{X: 0, Y: 5}, 5},
		{"Diagonal distance", Coord{X: 0, Y: 0}, Coord{X: 3, Y: 4}, 7},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := manhattanDistance(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Expected %d, got %d", tt.expected, result)
			}
		})
	}
}

// Test isImmediatelyFatal - out of bounds
func TestIsImmediatelyFatal_OutOfBounds(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{},
		},
	}

	tests := []struct {
		name     string
		pos      Coord
		expected bool
	}{
		{"Valid position", Coord{X: 5, Y: 5}, false},
		{"Out of bounds left", Coord{X: -1, Y: 5}, true},
		{"Out of bounds right", Coord{X: 11, Y: 5}, true},
		{"Out of bounds bottom", Coord{X: 5, Y: -1}, true},
		{"Out of bounds top", Coord{X: 5, Y: 11}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isImmediatelyFatal(state, tt.pos)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Test isImmediatelyFatal - snake collision
func TestIsImmediatelyFatal_SnakeCollision(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				{
					ID:     "snake1",
					Health: 50,
					Body: []Coord{
						{X: 5, Y: 5},
						{X: 5, Y: 4},
						{X: 5, Y: 3},
					},
				},
			},
		},
	}

	tests := []struct {
		name     string
		pos      Coord
		expected bool
	}{
		{"Empty space", Coord{X: 0, Y: 0}, false},
		{"Snake head", Coord{X: 5, Y: 5}, true},
		{"Snake body", Coord{X: 5, Y: 4}, true},
		{"Snake tail (should move)", Coord{X: 5, Y: 3}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isImmediatelyFatal(state, tt.pos)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v for position %v", tt.expected, result, tt.pos)
			}
		})
	}
}

// Test isImmediatelyFatal - snake just ate (tail won't move)
func TestIsImmediatelyFatal_SnakeJustAte(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				{
					ID:     "snake1",
					Health: MaxHealth, // Just ate
					Body: []Coord{
						{X: 5, Y: 5},
						{X: 5, Y: 4},
						{X: 5, Y: 3},
					},
				},
			},
		},
	}

	// Tail should be considered dangerous if snake just ate
	result := isImmediatelyFatal(state, Coord{X: 5, Y: 3})
	if !result {
		t.Error("Expected tail to be dangerous when snake just ate")
	}
}

// Test evaluateFoodProximity
func TestEvaluateFoodProximity(t *testing.T) {
	state := GameState{
		Board: Board{
			Food: []Coord{
				{X: 5, Y: 5},
				{X: 10, Y: 10},
			},
		},
	}

	tests := []struct {
		name     string
		pos      Coord
		hasValue bool
	}{
		{"At food location", Coord{X: 5, Y: 5}, true},
		{"Near food", Coord{X: 5, Y: 6}, true},
		{"Far from food", Coord{X: 0, Y: 0}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := evaluateFoodProximity(state, tt.pos)
			if tt.hasValue && result <= 0 {
				t.Errorf("Expected positive value, got %f", result)
			}
		})
	}
}

// Test evaluateFoodProximity with no food
func TestEvaluateFoodProximity_NoFood(t *testing.T) {
	state := GameState{
		Board: Board{
			Food: []Coord{},
		},
	}

	result := evaluateFoodProximity(state, Coord{X: 5, Y: 5})
	if result != 0 {
		t.Errorf("Expected 0 when no food, got %f", result)
	}
}

// Test evaluateHeadCollisionRisk
func TestEvaluateHeadCollisionRisk(t *testing.T) {
	state := GameState{
		You: Battlesnake{
			ID:     "me",
			Length: 5,
			Head:   Coord{X: 5, Y: 5},
		},
		Board: Board{
			Snakes: []Battlesnake{
				{
					ID:     "me",
					Length: 5,
					Head:   Coord{X: 5, Y: 5},
				},
				{
					ID:     "enemy1",
					Length: 6, // Larger snake
					Head:   Coord{X: 5, Y: 7},
				},
				{
					ID:     "enemy2",
					Length: 3, // Smaller snake
					Head:   Coord{X: 7, Y: 5},
				},
			},
		},
	}

	// Position where larger enemy could move to
	riskPos := Coord{X: 5, Y: 6}
	risk := evaluateHeadCollisionRisk(state, riskPos)
	
	if risk <= 0 {
		t.Errorf("Expected positive risk for position near larger enemy head, got %f", risk)
	}
}

// Test evaluateCenterProximity
func TestEvaluateCenterProximity(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
		},
	}

	centerScore := evaluateCenterProximity(state, Coord{X: 5, Y: 5})
	cornerScore := evaluateCenterProximity(state, Coord{X: 0, Y: 0})

	if centerScore <= cornerScore {
		t.Errorf("Expected center score (%f) to be higher than corner score (%f)", centerScore, cornerScore)
	}
}

// Test evaluateSpace
func TestEvaluateSpace(t *testing.T) {
	state := GameState{
		You: Battlesnake{
			Length: 3,
		},
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{},
		},
	}

	// Open space should have high score
	openSpaceScore := evaluateSpace(state, Coord{X: 5, Y: 5})
	if openSpaceScore <= 0 {
		t.Errorf("Expected positive score for open space, got %f", openSpaceScore)
	}
}

// Test move function
func TestMove(t *testing.T) {
	state := GameState{
		Turn: 1,
		Game: Game{
			ID: "test-game",
		},
		You: Battlesnake{
			ID:     "me",
			Health: 50,
			Length: 3,
			Head:   Coord{X: 5, Y: 5},
			Body: []Coord{
				{X: 5, Y: 5},
				{X: 5, Y: 4},
				{X: 5, Y: 3},
			},
		},
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 5, Y: 7}},
			Snakes: []Battlesnake{
				{
					ID:     "me",
					Health: 50,
					Length: 3,
					Head:   Coord{X: 5, Y: 5},
					Body: []Coord{
						{X: 5, Y: 5},
						{X: 5, Y: 4},
						{X: 5, Y: 3},
					},
				},
			},
		},
	}

	response := move(state)
	
	// Should return a valid move
	validMoves := map[string]bool{
		MoveUp:    true,
		MoveDown:  true,
		MoveLeft:  true,
		MoveRight: true,
	}
	
	if !validMoves[response.Move] {
		t.Errorf("Invalid move returned: %s", response.Move)
	}
	
	// Should not move down (into own body)
	if response.Move == MoveDown {
		t.Error("Should not move into own body")
	}
}

// Test scoreMove doesn't crash
func TestScoreMove(t *testing.T) {
	state := GameState{
		Turn: 1,
		You: Battlesnake{
			ID:     "me",
			Health: 50,
			Length: 3,
			Head:   Coord{X: 5, Y: 5},
			Body: []Coord{
				{X: 5, Y: 5},
				{X: 5, Y: 4},
				{X: 5, Y: 3},
			},
		},
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{},
			Snakes: []Battlesnake{
				{
					ID:     "me",
					Health: 50,
					Length: 3,
					Head:   Coord{X: 5, Y: 5},
					Body: []Coord{
						{X: 5, Y: 5},
						{X: 5, Y: 4},
						{X: 5, Y: 3},
					},
				},
			},
		},
	}

	// Test all moves
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	for _, move := range moves {
		score := scoreMove(state, move)
		// Fatal move should have very negative score
		if move == MoveDown {
			if score > -1000 {
				t.Errorf("Expected very negative score for fatal move down, got %f", score)
			}
		}
	}
}
