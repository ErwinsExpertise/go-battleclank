package board

import "testing"

func TestIsInBounds(t *testing.T) {
	b := &Board{Width: 11, Height: 11}
	
	tests := []struct {
		name     string
		pos      Coord
		expected bool
	}{
		{"Center position", Coord{X: 5, Y: 5}, true},
		{"Top-left corner", Coord{X: 0, Y: 0}, true},
		{"Bottom-right corner", Coord{X: 10, Y: 10}, true},
		{"Out of bounds left", Coord{X: -1, Y: 5}, false},
		{"Out of bounds right", Coord{X: 11, Y: 5}, false},
		{"Out of bounds top", Coord{X: 5, Y: 11}, false},
		{"Out of bounds bottom", Coord{X: 5, Y: -1}, false},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := b.IsInBounds(tt.pos)
			if result != tt.expected {
				t.Errorf("IsInBounds(%v) = %v, want %v", tt.pos, result, tt.expected)
			}
		})
	}
}

func TestIsOccupied(t *testing.T) {
	b := &Board{
		Width:  11,
		Height: 11,
		Snakes: []Snake{
			{
				ID:     "snake1",
				Health: 90, // Not just eaten, tail will move
				Body:   []Coord{{X: 5, Y: 5}, {X: 5, Y: 4}, {X: 5, Y: 3}},
				Head:   Coord{X: 5, Y: 5},
				Length: 3,
			},
		},
	}
	
	tests := []struct {
		name      string
		pos       Coord
		skipTails bool
		expected  bool
	}{
		{"Head position", Coord{X: 5, Y: 5}, false, true},
		{"Body position", Coord{X: 5, Y: 4}, false, true},
		{"Tail position, skip tails", Coord{X: 5, Y: 3}, true, false},
		{"Tail position, don't skip", Coord{X: 5, Y: 3}, false, true},
		{"Empty position", Coord{X: 6, Y: 6}, false, false},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := b.IsOccupied(tt.pos, tt.skipTails)
			if result != tt.expected {
				t.Errorf("IsOccupied(%v, %v) = %v, want %v", 
					tt.pos, tt.skipTails, result, tt.expected)
			}
		})
	}
}

func TestManhattanDistance(t *testing.T) {
	tests := []struct {
		name     string
		a        Coord
		b        Coord
		expected int
	}{
		{"Same position", Coord{X: 5, Y: 5}, Coord{X: 5, Y: 5}, 0},
		{"Horizontal distance", Coord{X: 0, Y: 0}, Coord{X: 5, Y: 0}, 5},
		{"Vertical distance", Coord{X: 0, Y: 0}, Coord{X: 0, Y: 5}, 5},
		{"Diagonal distance", Coord{X: 0, Y: 0}, Coord{X: 3, Y: 4}, 7},
		{"Negative coordinates", Coord{X: -2, Y: -2}, Coord{X: 3, Y: 4}, 11},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ManhattanDistance(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("ManhattanDistance(%v, %v) = %v, want %v", 
					tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestGetNeighbors(t *testing.T) {
	b := &Board{Width: 11, Height: 11}
	
	tests := []struct {
		name     string
		pos      Coord
		expected int // number of neighbors
	}{
		{"Center position", Coord{X: 5, Y: 5}, 4},
		{"Top-left corner", Coord{X: 0, Y: 0}, 2},
		{"Bottom-right corner", Coord{X: 10, Y: 10}, 2},
		{"Edge position", Coord{X: 0, Y: 5}, 3},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			neighbors := b.GetNeighbors(tt.pos)
			if len(neighbors) != tt.expected {
				t.Errorf("GetNeighbors(%v) returned %d neighbors, want %d", 
					tt.pos, len(neighbors), tt.expected)
			}
		})
	}
}

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
			result := GetNextPosition(tt.pos, tt.move)
			if result != tt.expected {
				t.Errorf("GetNextPosition(%v, %s) = %v, want %v", 
					tt.pos, tt.move, result, tt.expected)
			}
		})
	}
}
