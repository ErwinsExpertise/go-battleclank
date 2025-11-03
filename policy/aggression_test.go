package policy

import (
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"testing"
)

func TestCalculateAggressionScore(t *testing.T) {
	tests := []struct {
		name        string
		state       *board.GameState
		mySpace     float64
		minScore    float64
		maxScore    float64
		description string
	}{
		{
			name: "Healthy and dominant",
			state: &board.GameState{
				Board: board.Board{
					Width:  11,
					Height: 11,
					Snakes: []board.Snake{
						{ID: "you", Health: 80, Length: 10, Head: board.Coord{X: 5, Y: 5}},
						{ID: "enemy1", Health: 60, Length: 5, Head: board.Coord{X: 1, Y: 1}},
					},
				},
				You: board.Snake{ID: "you", Health: 80, Length: 10, Head: board.Coord{X: 5, Y: 5}},
			},
			mySpace:     0.5,
			minScore:    0.7,
			maxScore:    1.0,
			description: "Should be aggressive when healthy and dominant",
		},
		{
			name: "Critical health",
			state: &board.GameState{
				Board: board.Board{
					Width:  11,
					Height: 11,
					Snakes: []board.Snake{
						{ID: "you", Health: 20, Length: 5, Head: board.Coord{X: 5, Y: 5}},
						{ID: "enemy1", Health: 60, Length: 5, Head: board.Coord{X: 1, Y: 1}},
					},
				},
				You: board.Snake{ID: "you", Health: 20, Length: 5, Head: board.Coord{X: 5, Y: 5}},
			},
			mySpace:     0.3,
			minScore:    0.0,
			maxScore:    0.4,
			description: "Should be defensive at critical health",
		},
		{
			name: "Outmatched",
			state: &board.GameState{
				Board: board.Board{
					Width:  11,
					Height: 11,
					Snakes: []board.Snake{
						{ID: "you", Health: 60, Length: 3, Head: board.Coord{X: 5, Y: 5}},
						{ID: "enemy1", Health: 80, Length: 10, Head: board.Coord{X: 6, Y: 6}},
					},
				},
				You: board.Snake{ID: "you", Health: 60, Length: 3, Head: board.Coord{X: 5, Y: 5}},
			},
			mySpace:     0.3,
			minScore:    0.0,
			maxScore:    0.5,
			description: "Should be defensive when outmatched",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CalculateAggressionScore(tt.state, tt.mySpace)
			
			if result.Score < tt.minScore || result.Score > tt.maxScore {
				t.Errorf("%s: got score %.2f, want between %.2f and %.2f",
					tt.description, result.Score, tt.minScore, tt.maxScore)
			}
			
			t.Logf("✓ %s: Aggression score %.2f (in range %.2f-%.2f)",
				tt.description, result.Score, tt.minScore, tt.maxScore)
		})
	}
}

func TestShouldAttemptTrap(t *testing.T) {
	tests := []struct {
		name     string
		score    AggressionScore
		expected bool
	}{
		{"High aggression", AggressionScore{Score: 0.8}, true},
		{"Moderate aggression", AggressionScore{Score: 0.5}, false},
		{"Low aggression", AggressionScore{Score: 0.2}, false},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ShouldAttemptTrap(tt.score)
			if result != tt.expected {
				t.Errorf("ShouldAttemptTrap(%.2f) = %v, want %v",
					tt.score.Score, result, tt.expected)
			}
		})
	}
}

func TestShouldPrioritizeSurvival(t *testing.T) {
	tests := []struct {
		name     string
		score    AggressionScore
		expected bool
	}{
		{"High aggression", AggressionScore{Score: 0.8}, false},
		{"Moderate aggression", AggressionScore{Score: 0.5}, false},
		{"Low aggression", AggressionScore{Score: 0.2}, true},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ShouldPrioritizeSurvival(tt.score)
			if result != tt.expected {
				t.Errorf("ShouldPrioritizeSurvival(%.2f) = %v, want %v",
					tt.score.Score, result, tt.expected)
			}
		})
	}
}

func TestGetFoodWeight(t *testing.T) {
	tests := []struct {
		name       string
		health     int
		outmatched bool
		minWeight  float64
		maxWeight  float64
	}{
		{"Critical health, outmatched", 20, true, 380, 420},      // Reduced: 400
		{"Critical health, not outmatched", 20, false, 480, 520}, // Reduced: 500
		{"Low health, outmatched", 40, true, 170, 190},           // Reduced: 180
		{"Low health, not outmatched", 40, false, 210, 230},      // Reduced: 220
		{"Healthy, outmatched", 80, true, 8, 12},                 // NEW: Health ceiling at 80+ = 10.0
		{"Healthy, not outmatched", 80, false, 8, 12},            // NEW: Health ceiling at 80+ = 10.0
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := &board.GameState{
				You: board.Snake{Health: tt.health},
			}
			
			aggression := AggressionScore{Score: 0.5}
			weight := GetFoodWeight(state, aggression, tt.outmatched)
			
			if weight < tt.minWeight || weight > tt.maxWeight {
				t.Errorf("GetFoodWeight(health=%d, outmatched=%v) = %.2f, want between %.2f and %.2f",
					tt.health, tt.outmatched, weight, tt.minWeight, tt.maxWeight)
			}
		})
	}
}

func TestIsOutmatched(t *testing.T) {
	tests := []struct {
		name     string
		state    *board.GameState
		expected bool
	}{
		{
			name: "Enemy much larger and nearby",
			state: &board.GameState{
				Board: board.Board{
					Width:  11,
					Height: 11,
					Snakes: []board.Snake{
						{ID: "you", Length: 3, Head: board.Coord{X: 5, Y: 5}},
						{ID: "enemy1", Length: 8, Head: board.Coord{X: 6, Y: 6}},
					},
				},
				You: board.Snake{ID: "you", Length: 3, Head: board.Coord{X: 5, Y: 5}},
			},
			expected: true,
		},
		{
			name: "Enemy slightly larger",
			state: &board.GameState{
				Board: board.Board{
					Width:  11,
					Height: 11,
					Snakes: []board.Snake{
						{ID: "you", Length: 5, Head: board.Coord{X: 5, Y: 5}},
						{ID: "enemy1", Length: 7, Head: board.Coord{X: 6, Y: 6}},
					},
				},
				You: board.Snake{ID: "you", Length: 5, Head: board.Coord{X: 5, Y: 5}},
			},
			expected: false,
		},
		{
			name: "Enemy much larger but far away",
			state: &board.GameState{
				Board: board.Board{
					Width:  11,
					Height: 11,
					Snakes: []board.Snake{
						{ID: "you", Length: 3, Head: board.Coord{X: 0, Y: 0}},
						{ID: "enemy1", Length: 10, Head: board.Coord{X: 10, Y: 10}},
					},
				},
				You: board.Snake{ID: "you", Length: 3, Head: board.Coord{X: 0, Y: 0}},
			},
			expected: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsOutmatched(tt.state, 3)
			if result != tt.expected {
				t.Errorf("IsOutmatched() = %v, want %v", result, tt.expected)
			}
		})
	}
}
