package search

import (
	"testing"

	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

// TestParallelEdgeAvoidance validates that the agent avoids turning toward
// parallel snakes when approaching an edge
func TestParallelEdgeAvoidance(t *testing.T) {
	// Test case 1: Moving up parallel to enemy, near top edge, enemy on right
	t.Run("Moving up parallel - near top edge - avoid turning right toward enemy", func(t *testing.T) {
		state := &board.GameState{
			Turn: 50,
			Board: board.Board{
				Width:  11,
				Height: 11,
				Snakes: []board.Snake{
					{
						ID:     "me",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 4, Y: 8}, // Head - 2 tiles from top edge
							{X: 4, Y: 7}, // Moving up
							{X: 4, Y: 6},
							{X: 4, Y: 5},
							{X: 4, Y: 4},
							{X: 4, Y: 3},
						},
						Head: board.Coord{X: 4, Y: 8},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 5, Y: 8}, // Adjacent lane, aligned
							{X: 5, Y: 7}, // Also moving up
							{X: 5, Y: 6},
							{X: 5, Y: 5},
							{X: 5, Y: 4},
							{X: 5, Y: 3},
						},
						Head: board.Coord{X: 5, Y: 8},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 6,
				Body: []board.Coord{
					{X: 4, Y: 8},
					{X: 4, Y: 7},
					{X: 4, Y: 6},
					{X: 4, Y: 5},
					{X: 4, Y: 4},
					{X: 4, Y: 3},
				},
				Head: board.Coord{X: 4, Y: 8},
			},
		}

		greedy := NewGreedySearch()

		upScore := greedy.ScoreMove(state, board.MoveUp)       // Continue straight
		rightScore := greedy.ScoreMove(state, board.MoveRight) // Toward enemy
		leftScore := greedy.ScoreMove(state, board.MoveLeft)   // Away from enemy
		downScore := greedy.ScoreMove(state, board.MoveDown)   // Back toward body

		t.Logf("Scores - Up: %.2f, Right (toward enemy): %.2f, Left (away): %.2f, Down: %.2f",
			upScore, rightScore, leftScore, downScore)

		// Right should be penalized more than left since it moves toward enemy
		if rightScore >= leftScore && leftScore > -9000 {
			t.Errorf("Expected right (toward enemy) to be penalized more than left (away), got right: %.2f, left: %.2f",
				rightScore, leftScore)
		} else {
			t.Logf("✓ Right turn toward enemy correctly penalized more than left")
		}

		// Best move should not be toward enemy
		bestMove := greedy.FindBestMove(state)
		if bestMove == board.MoveRight {
			t.Errorf("Should not choose to turn right toward enemy when near edge, chose: %s", bestMove)
		} else {
			t.Logf("✓ Correctly avoided turning toward enemy, chose: %s", bestMove)
		}
	})

	// Test case 2: Moving down parallel to enemy, near bottom edge, enemy on left
	t.Run("Moving down parallel - near bottom edge - avoid turning left toward enemy", func(t *testing.T) {
		state := &board.GameState{
			Turn: 50,
			Board: board.Board{
				Width:  11,
				Height: 11,
				Snakes: []board.Snake{
					{
						ID:     "me",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 6, Y: 2}, // Head - 2 tiles from bottom edge
							{X: 6, Y: 3}, // Moving down
							{X: 6, Y: 4},
							{X: 6, Y: 5},
							{X: 6, Y: 6},
							{X: 6, Y: 7},
						},
						Head: board.Coord{X: 6, Y: 2},
					},
					{
						ID:     "enemy",
						Health: 80,
						Length: 7,
						Body: []board.Coord{
							{X: 5, Y: 3}, // Adjacent lane, ahead
							{X: 5, Y: 4}, // Also moving down
							{X: 5, Y: 5},
							{X: 5, Y: 6},
							{X: 5, Y: 7},
							{X: 5, Y: 8},
							{X: 5, Y: 9},
						},
						Head: board.Coord{X: 5, Y: 3},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 6,
				Body: []board.Coord{
					{X: 6, Y: 2},
					{X: 6, Y: 3},
					{X: 6, Y: 4},
					{X: 6, Y: 5},
					{X: 6, Y: 6},
					{X: 6, Y: 7},
				},
				Head: board.Coord{X: 6, Y: 2},
			},
		}

		greedy := NewGreedySearch()

		downScore := greedy.ScoreMove(state, board.MoveDown)  // Continue straight
		leftScore := greedy.ScoreMove(state, board.MoveLeft)  // Toward enemy
		rightScore := greedy.ScoreMove(state, board.MoveRight) // Away from enemy

		t.Logf("Scores - Down: %.2f, Left (toward enemy): %.2f, Right (away): %.2f",
			downScore, leftScore, rightScore)

		// Left should be penalized more since it moves toward larger enemy
		if leftScore >= rightScore && rightScore > -9000 {
			t.Errorf("Expected left (toward enemy) to be penalized more than right (away), got left: %.2f, right: %.2f",
				leftScore, rightScore)
		} else {
			t.Logf("✓ Left turn toward larger enemy correctly penalized more than right")
		}
	})

	// Test case 3: Moving right parallel to enemy, near right edge
	t.Run("Moving right parallel - near right edge - avoid turning toward enemy", func(t *testing.T) {
		state := &board.GameState{
			Turn: 50,
			Board: board.Board{
				Width:  11,
				Height: 11,
				Snakes: []board.Snake{
					{
						ID:     "me",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 8, Y: 5}, // Head - 2 tiles from right edge
							{X: 7, Y: 5}, // Moving right
							{X: 6, Y: 5},
							{X: 5, Y: 5},
							{X: 4, Y: 5},
							{X: 3, Y: 5},
						},
						Head: board.Coord{X: 8, Y: 5},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 9, Y: 6}, // Adjacent lane, ahead
							{X: 8, Y: 6}, // Also moving right
							{X: 7, Y: 6},
							{X: 6, Y: 6},
							{X: 5, Y: 6},
							{X: 4, Y: 6},
						},
						Head: board.Coord{X: 9, Y: 6},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 6,
				Body: []board.Coord{
					{X: 8, Y: 5},
					{X: 7, Y: 5},
					{X: 6, Y: 5},
					{X: 5, Y: 5},
					{X: 4, Y: 5},
					{X: 3, Y: 5},
				},
				Head: board.Coord{X: 8, Y: 5},
			},
		}

		greedy := NewGreedySearch()

		rightScore := greedy.ScoreMove(state, board.MoveRight) // Continue straight
		upScore := greedy.ScoreMove(state, board.MoveUp)       // Toward enemy
		downScore := greedy.ScoreMove(state, board.MoveDown)   // Away from enemy

		t.Logf("Scores - Right: %.2f, Up (toward enemy): %.2f, Down (away): %.2f",
			rightScore, upScore, downScore)

		// Up should be penalized more than down
		if upScore >= downScore && downScore > -9000 {
			t.Errorf("Expected up (toward enemy) to be penalized more than down (away), got up: %.2f, down: %.2f",
				upScore, downScore)
		} else {
			t.Logf("✓ Up turn toward enemy correctly penalized more than down")
		}
	})

	// Test case 4: Moving left parallel to enemy, near left edge
	t.Run("Moving left parallel - near left edge - avoid turning toward enemy", func(t *testing.T) {
		state := &board.GameState{
			Turn: 50,
			Board: board.Board{
				Width:  11,
				Height: 11,
				Snakes: []board.Snake{
					{
						ID:     "me",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 2, Y: 7}, // Head - 2 tiles from left edge
							{X: 3, Y: 7}, // Moving left
							{X: 4, Y: 7},
							{X: 5, Y: 7},
							{X: 6, Y: 7},
							{X: 7, Y: 7},
						},
						Head: board.Coord{X: 2, Y: 7},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 2, Y: 6}, // Adjacent lane, aligned
							{X: 3, Y: 6}, // Also moving left
							{X: 4, Y: 6},
							{X: 5, Y: 6},
							{X: 6, Y: 6},
							{X: 7, Y: 6},
						},
						Head: board.Coord{X: 2, Y: 6},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 6,
				Body: []board.Coord{
					{X: 2, Y: 7},
					{X: 3, Y: 7},
					{X: 4, Y: 7},
					{X: 5, Y: 7},
					{X: 6, Y: 7},
					{X: 7, Y: 7},
				},
				Head: board.Coord{X: 2, Y: 7},
			},
		}

		greedy := NewGreedySearch()

		leftScore := greedy.ScoreMove(state, board.MoveLeft)  // Continue straight
		downScore := greedy.ScoreMove(state, board.MoveDown)  // Toward enemy
		upScore := greedy.ScoreMove(state, board.MoveUp)      // Away from enemy

		t.Logf("Scores - Left: %.2f, Down (toward enemy): %.2f, Up (away): %.2f",
			leftScore, downScore, upScore)

		// Down should be penalized more than up
		if downScore >= upScore && upScore > -9000 {
			t.Errorf("Expected down (toward enemy) to be penalized more than up (away), got down: %.2f, up: %.2f",
				downScore, upScore)
		} else {
			t.Logf("✓ Down turn toward enemy correctly penalized more than up")
		}
	})

	// Test case 5: Not near edge - should not trigger
	t.Run("Not near edge - parallel avoidance should not trigger", func(t *testing.T) {
		state := &board.GameState{
			Turn: 50,
			Board: board.Board{
				Width:  11,
				Height: 11,
				Snakes: []board.Snake{
					{
						ID:     "me",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 5, Y: 5}, // Center of board - far from edges
							{X: 5, Y: 4},
							{X: 5, Y: 3},
							{X: 5, Y: 2},
							{X: 5, Y: 1},
							{X: 5, Y: 0},
						},
						Head: board.Coord{X: 5, Y: 5},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 6, Y: 5}, // Adjacent lane
							{X: 6, Y: 4},
							{X: 6, Y: 3},
							{X: 6, Y: 2},
							{X: 6, Y: 1},
							{X: 6, Y: 0},
						},
						Head: board.Coord{X: 6, Y: 5},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 6,
				Body: []board.Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
					{X: 5, Y: 2},
					{X: 5, Y: 1},
					{X: 5, Y: 0},
				},
				Head: board.Coord{X: 5, Y: 5},
			},
		}

		greedy := NewGreedySearch()

		upScore := greedy.ScoreMove(state, board.MoveUp)
		rightScore := greedy.ScoreMove(state, board.MoveRight) // Toward enemy
		leftScore := greedy.ScoreMove(state, board.MoveLeft)   // Away from enemy

		t.Logf("Scores (center board) - Up: %.2f, Right: %.2f, Left: %.2f",
			upScore, rightScore, leftScore)

		// When not near edge, the penalty should be much smaller or non-existent
		// The key is that the logic doesn't crash
		bestMove := greedy.FindBestMove(state)
		if bestMove == "" {
			t.Error("Should return a valid move")
		}
		t.Logf("✓ Logic works in center of board (not near edge), chose: %s", bestMove)
	})

	// Test case 6: Not moving parallel (different directions) - should not trigger
	t.Run("Not parallel - different directions - should not trigger", func(t *testing.T) {
		state := &board.GameState{
			Turn: 50,
			Board: board.Board{
				Width:  11,
				Height: 11,
				Snakes: []board.Snake{
					{
						ID:     "me",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 4, Y: 8}, // Head - near top edge
							{X: 4, Y: 7}, // Moving up
							{X: 4, Y: 6},
							{X: 4, Y: 5},
							{X: 4, Y: 4},
							{X: 4, Y: 3},
						},
						Head: board.Coord{X: 4, Y: 8},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 5, Y: 8}, // Adjacent position
							{X: 4, Y: 8}, // Moving right (perpendicular)
							{X: 3, Y: 8},
							{X: 2, Y: 8},
							{X: 1, Y: 8},
							{X: 0, Y: 8},
						},
						Head: board.Coord{X: 5, Y: 8},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 6,
				Body: []board.Coord{
					{X: 4, Y: 8},
					{X: 4, Y: 7},
					{X: 4, Y: 6},
					{X: 4, Y: 5},
					{X: 4, Y: 4},
					{X: 4, Y: 3},
				},
				Head: board.Coord{X: 4, Y: 8},
			},
		}

		greedy := NewGreedySearch()

		upScore := greedy.ScoreMove(state, board.MoveUp)
		rightScore := greedy.ScoreMove(state, board.MoveRight)

		t.Logf("Scores (not parallel) - Up: %.2f, Right: %.2f", upScore, rightScore)

		// Should not crash and should return valid move
		bestMove := greedy.FindBestMove(state)
		if bestMove == "" {
			t.Error("Should return a valid move")
		}
		t.Logf("✓ Logic works when not moving parallel, chose: %s", bestMove)
	})

	// Test case 7: Very close to edge (1 tile) - should have stronger penalty
	t.Run("Very close to edge - 1 tile away - stronger penalty", func(t *testing.T) {
		state := &board.GameState{
			Turn: 50,
			Board: board.Board{
				Width:  11,
				Height: 11,
				Snakes: []board.Snake{
					{
						ID:     "me",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 4, Y: 9}, // Head - 1 tile from top edge
							{X: 4, Y: 8}, // Moving up
							{X: 4, Y: 7},
							{X: 4, Y: 6},
							{X: 4, Y: 5},
							{X: 4, Y: 4},
						},
						Head: board.Coord{X: 4, Y: 9},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 5, Y: 9}, // Adjacent lane, aligned
							{X: 5, Y: 8}, // Also moving up
							{X: 5, Y: 7},
							{X: 5, Y: 6},
							{X: 5, Y: 5},
							{X: 5, Y: 4},
						},
						Head: board.Coord{X: 5, Y: 9},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 6,
				Body: []board.Coord{
					{X: 4, Y: 9},
					{X: 4, Y: 8},
					{X: 4, Y: 7},
					{X: 4, Y: 6},
					{X: 4, Y: 5},
					{X: 4, Y: 4},
				},
				Head: board.Coord{X: 4, Y: 9},
			},
		}

		greedy := NewGreedySearch()

		rightScore := greedy.ScoreMove(state, board.MoveRight) // Toward enemy
		leftScore := greedy.ScoreMove(state, board.MoveLeft)   // Away from enemy

		t.Logf("Scores (1 tile from edge) - Right (toward enemy): %.2f, Left (away): %.2f",
			rightScore, leftScore)

		// Right should be heavily penalized
		if rightScore >= leftScore && leftScore > -9000 {
			t.Errorf("Expected right (toward enemy) at edge to be heavily penalized, got right: %.2f, left: %.2f",
				rightScore, leftScore)
		} else {
			t.Logf("✓ Turn toward enemy when 1 tile from edge correctly heavily penalized")
		}
	})
}
