package search

import (
	"testing"

	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

// TestEmergencyWallEscape tests the specific scenario from the issue:
// Snake 1 tile from wall, enemy head-on within 2 tiles, turning away causes collision
func TestEmergencyWallEscape(t *testing.T) {
	t.Run("Enemy head-on - turning away from wall causes head-to-head", func(t *testing.T) {
		// Snake at X=1 (1 tile from left wall), moving right toward X=2
		// Enemy at X=3, moving left
		// If snake continues right (away from wall), head-on collision at X=2
		// If snake turns left (toward wall), goes to X=0 (wall) - safe
		// Perpendicular options should be evaluated but wall turn should be considered
		state := &board.GameState{
			Turn: 50,
			Board: board.Board{
				Width:  11,
				Height: 11,
				Snakes: []board.Snake{
					{
						ID:     "me",
						Health: 70,
						Length: 5,
						Body: []board.Coord{
							{X: 1, Y: 5}, // Head at X=1
							{X: 0, Y: 5}, // Came from wall
							{X: 0, Y: 4},
							{X: 0, Y: 3},
							{X: 0, Y: 2},
						},
						Head: board.Coord{X: 1, Y: 5},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 6, // Equal or larger - head-on is bad for us
						Body: []board.Coord{
							{X: 3, Y: 5}, // 2 tiles away
							{X: 4, Y: 5}, // Moving toward us
							{X: 5, Y: 5},
							{X: 6, Y: 5},
							{X: 7, Y: 5},
							{X: 8, Y: 5},
						},
						Head: board.Coord{X: 3, Y: 5},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 5,
				Body: []board.Coord{
					{X: 1, Y: 5},
					{X: 0, Y: 5},
					{X: 0, Y: 4},
					{X: 0, Y: 3},
					{X: 0, Y: 2},
				},
				Head: board.Coord{X: 1, Y: 5},
			},
		}

		greedy := NewGreedySearch()

		leftScore := greedy.ScoreMove(state, board.MoveLeft)   // Back toward wall
		rightScore := greedy.ScoreMove(state, board.MoveRight) // Away from wall toward enemy
		upScore := greedy.ScoreMove(state, board.MoveUp)       // Perpendicular
		downScore := greedy.ScoreMove(state, board.MoveDown)   // Perpendicular

		t.Logf("Scores - Left (wall): %.2f, Right (toward enemy): %.2f, Up: %.2f, Down: %.2f",
			leftScore, rightScore, upScore, downScore)

		bestMove := greedy.FindBestMove(state)
		t.Logf("Best move: %s", bestMove)

		// Right should be heavily penalized - it's moving toward a head-on collision
		// with an equal/larger snake
		if rightScore > -500 {
			t.Errorf("Right (toward enemy) should have heavy penalty but scored %.2f", rightScore)
		}

		// Best move should definitely NOT be right (toward enemy)
		if bestMove == board.MoveRight {
			t.Errorf("CRITICAL BUG: Snake chose right (toward enemy head-on collision)!")
		}
	})

	t.Run("Wall adjacent - enemy 1 tile away - emergency turn", func(t *testing.T) {
		// Most critical scenario: we're AT the wall (X=0), enemy is 1 tile away (X=1)
		// We can only move right (into enemy path), up, or down
		// This tests if we detect the immediate danger correctly
		state := &board.GameState{
			Turn: 50,
			Board: board.Board{
				Width:  11,
				Height: 11,
				Snakes: []board.Snake{
					{
						ID:     "me",
						Health: 70,
						Length: 5,
						Body: []board.Coord{
							{X: 0, Y: 5}, // At wall
							{X: 0, Y: 4}, // Moving up along wall
							{X: 0, Y: 3},
							{X: 1, Y: 3},
							{X: 1, Y: 2},
						},
						Head: board.Coord{X: 0, Y: 5},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 7, // Larger - head-on is fatal
						Body: []board.Coord{
							{X: 1, Y: 5}, // 1 tile away - IMMEDIATE danger
							{X: 2, Y: 5},
							{X: 3, Y: 5},
							{X: 4, Y: 5},
							{X: 5, Y: 5},
							{X: 6, Y: 5},
							{X: 7, Y: 5},
						},
						Head: board.Coord{X: 1, Y: 5},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 5,
				Body: []board.Coord{
					{X: 0, Y: 5},
					{X: 0, Y: 4},
					{X: 0, Y: 3},
					{X: 1, Y: 3},
					{X: 1, Y: 2},
				},
				Head: board.Coord{X: 0, Y: 5},
			},
		}

		greedy := NewGreedySearch()

		leftScore := greedy.ScoreMove(state, board.MoveLeft)   // Out of bounds
		rightScore := greedy.ScoreMove(state, board.MoveRight) // Into enemy's position
		upScore := greedy.ScoreMove(state, board.MoveUp)       // Safe perpendicular
		downScore := greedy.ScoreMove(state, board.MoveDown)   // Safe perpendicular

		t.Logf("Scores - Left (OOB): %.2f, Right (enemy pos): %.2f, Up: %.2f, Down: %.2f",
			leftScore, rightScore, upScore, downScore)

		bestMove := greedy.FindBestMove(state)
		t.Logf("Best move: %s", bestMove)

		// Right should be fatal (enemy occupies that space)
		if rightScore > -5000 {
			t.Errorf("Right should be fatal but scored %.2f", rightScore)
		}

		// Must choose up or down
		if bestMove != board.MoveUp && bestMove != board.MoveDown {
			t.Errorf("Should choose perpendicular move but chose: %s", bestMove)
		}
	})

	t.Run("Both perpendiculars have enemies - wall turn is safest", func(t *testing.T) {
		// Complex scenario: near wall, enemy head-on, AND enemies on perpendicular paths
		// Wall direction (even constrained) becomes the safest option
		state := &board.GameState{
			Turn: 50,
			Board: board.Board{
				Width:  11,
				Height: 11,
				Snakes: []board.Snake{
					{
						ID:     "me",
						Health: 70,
						Length: 5,
						Body: []board.Coord{
							{X: 1, Y: 5}, // 1 tile from left wall
							{X: 2, Y: 5},
							{X: 3, Y: 5},
							{X: 4, Y: 5},
							{X: 5, Y: 5},
						},
						Head: board.Coord{X: 1, Y: 5},
					},
					{
						ID:     "enemy1",
						Health: 70,
						Length: 6,
						Body: []board.Coord{
							{X: 3, Y: 5}, // Head-on threat
							{X: 4, Y: 5},
							{X: 5, Y: 5},
							{X: 6, Y: 5},
							{X: 7, Y: 5},
							{X: 8, Y: 5},
						},
						Head: board.Coord{X: 3, Y: 5},
					},
					{
						ID:     "enemy2",
						Health: 70,
						Length: 5,
						Body: []board.Coord{
							{X: 2, Y: 7}, // Threatens up direction
							{X: 3, Y: 7},
							{X: 4, Y: 7},
							{X: 5, Y: 7},
							{X: 6, Y: 7},
						},
						Head: board.Coord{X: 2, Y: 7},
					},
					{
						ID:     "enemy3",
						Health: 70,
						Length: 5,
						Body: []board.Coord{
							{X: 2, Y: 3}, // Threatens down direction
							{X: 3, Y: 3},
							{X: 4, Y: 3},
							{X: 5, Y: 3},
							{X: 6, Y: 3},
						},
						Head: board.Coord{X: 2, Y: 3},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 5,
				Body: []board.Coord{
					{X: 1, Y: 5},
					{X: 2, Y: 5},
					{X: 3, Y: 5},
					{X: 4, Y: 5},
					{X: 5, Y: 5},
				},
				Head: board.Coord{X: 1, Y: 5},
			},
		}

		greedy := NewGreedySearch()

		leftScore := greedy.ScoreMove(state, board.MoveLeft)   // Toward wall (safest?)
		rightScore := greedy.ScoreMove(state, board.MoveRight) // Head-on collision
		upScore := greedy.ScoreMove(state, board.MoveUp)       // Enemy nearby
		downScore := greedy.ScoreMove(state, board.MoveDown)   // Enemy nearby

		t.Logf("Complex scenario scores:")
		t.Logf("  Left (wall): %.2f", leftScore)
		t.Logf("  Right (head-on): %.2f", rightScore)
		t.Logf("  Up (enemy): %.2f", upScore)
		t.Logf("  Down (enemy): %.2f", downScore)

		bestMove := greedy.FindBestMove(state)
		t.Logf("Best move: %s", bestMove)

		// Right should be worst (fatal head-on)
		if rightScore > leftScore || rightScore > upScore || rightScore > downScore {
			t.Errorf("Right (head-on) should be worst move but scored %.2f", rightScore)
		}

		// This scenario tests that when surrounded, we pick the least dangerous option
		// Even if wall is constraining, it's better than guaranteed collision
	})
}
