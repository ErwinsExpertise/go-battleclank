package search

import (
	"testing"

	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

// TestStraightMovementBonus verifies that the snake prefers to continue straight
// when there's no compelling reason to turn
func TestStraightMovementBonus(t *testing.T) {
	tests := []struct {
		name         string
		state        *board.GameState
		expectedMove string
		description  string
	}{
		{
			name: "Continue straight when healthy and no obstacles",
			state: &board.GameState{
				Turn: 30,
				Board: board.Board{
					Height: 11,
					Width:  11,
					Food: []board.Coord{
						{X: 5, Y: 8}, // food perpendicular to movement
					},
					Snakes: []board.Snake{
						{
							ID:     "me",
							Health: 70,
							Body: []board.Coord{
								{X: 5, Y: 5}, // head
								{X: 4, Y: 5}, // moving right
								{X: 3, Y: 5},
							},
							Head:   board.Coord{X: 5, Y: 5},
							Length: 3,
						},
					},
				},
				You: board.Snake{
					ID:     "me",
					Health: 70,
					Body: []board.Coord{
						{X: 5, Y: 5},
						{X: 4, Y: 5},
						{X: 3, Y: 5},
					},
					Head:   board.Coord{X: 5, Y: 5},
					Length: 3,
				},
			},
			expectedMove: "right",
			description:  "Should continue straight when healthy and no immediate needs",
		},
		{
			name: "Turn for critical health food",
			state: &board.GameState{
				Turn: 30,
				Board: board.Board{
					Height: 11,
					Width:  11,
					Food: []board.Coord{
						{X: 5, Y: 7}, // food perpendicular and close
					},
					Snakes: []board.Snake{
						{
							ID:     "me",
							Health: 15, // critical!
							Body: []board.Coord{
								{X: 5, Y: 5}, // head
								{X: 4, Y: 5}, // moving right
								{X: 3, Y: 5},
							},
							Head:   board.Coord{X: 5, Y: 5},
							Length: 3,
						},
					},
				},
				You: board.Snake{
					ID:     "me",
					Health: 15,
					Body: []board.Coord{
						{X: 5, Y: 5},
						{X: 4, Y: 5},
						{X: 3, Y: 5},
					},
					Head:   board.Coord{X: 5, Y: 5},
					Length: 3,
				},
			},
			expectedMove: "up",
			description:  "Should turn for food when health is critical despite straight bonus",
		},
		{
			name: "Continue straight in open space",
			state: &board.GameState{
				Turn: 20,
				Board: board.Board{
					Height: 11,
					Width:  11,
					Food: []board.Coord{
						{X: 8, Y: 2}, // food not aligned
					},
					Snakes: []board.Snake{
						{
							ID:     "me",
							Health: 80,
							Body: []board.Coord{
								{X: 5, Y: 5}, // head
								{X: 5, Y: 4}, // moving up
								{X: 5, Y: 3},
							},
							Head:   board.Coord{X: 5, Y: 5},
							Length: 3,
						},
					},
				},
				You: board.Snake{
					ID:     "me",
					Health: 80,
					Body: []board.Coord{
						{X: 5, Y: 5},
						{X: 5, Y: 4},
						{X: 5, Y: 3},
					},
					Head:   board.Coord{X: 5, Y: 5},
					Length: 3,
				},
			},
			expectedMove: "up",
			description:  "Should continue upward when healthy and space is open",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := NewGreedySearch()
			bestMove := strategy.FindBestMove(tt.state)

			if bestMove != tt.expectedMove {
				t.Errorf("%s: got %s, want %s", tt.description, bestMove, tt.expectedMove)

				// Log all move scores for debugging
				t.Logf("Move scores:")
				for _, move := range board.AllMoves() {
					score := strategy.ScoreMove(tt.state, move)
					if score > -1000 {
						t.Logf("  %s: %.2f", move, score)
					}
				}
			} else {
				t.Logf("✓ %s", tt.description)
			}
		})
	}
}
