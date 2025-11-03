package search

import (
	"testing"

	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

// TestWallUTurnDefense tests the specific scenario from the issue:
// Snake moving up near left wall, enemy moving down, both at similar Y positions
// Snake should turn into wall for a U-turn rather than turning away and colliding
func TestWallUTurnDefense(t *testing.T) {
	t.Run("Moving up near left wall - enemy approaching from ahead - should turn left into wall", func(t *testing.T) {
		// Our snake: at X=1, Y=5, moving UP (came from Y=4)
		// Enemy snake: at X=3, Y=7, moving DOWN (toward Y=6)
		// If we turn RIGHT (to X=2), enemy can turn right (to X=2) and intercept us
		// If we turn LEFT (to X=0, the wall), we can U-turn safely
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
							{X: 1, Y: 5}, // Head - moving up
							{X: 1, Y: 4}, // Came from below
							{X: 1, Y: 3},
							{X: 1, Y: 2},
							{X: 1, Y: 1},
						},
						Head: board.Coord{X: 1, Y: 5},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 5, // Same size - head-on is mutually fatal
						Body: []board.Coord{
							{X: 3, Y: 7}, // Head - moving down, 2 tiles away vertically
							{X: 3, Y: 8}, // Came from above
							{X: 3, Y: 9},
							{X: 4, Y: 9},
							{X: 5, Y: 9},
						},
						Head: board.Coord{X: 3, Y: 7},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 5,
				Body: []board.Coord{
					{X: 1, Y: 5},
					{X: 1, Y: 4},
					{X: 1, Y: 3},
					{X: 1, Y: 2},
					{X: 1, Y: 1},
				},
				Head: board.Coord{X: 1, Y: 5},
			},
		}

		greedy := NewGreedySearch()

		leftScore := greedy.ScoreMove(state, board.MoveLeft)   // Into wall (safe U-turn option)
		rightScore := greedy.ScoreMove(state, board.MoveRight) // Away from wall (dangerous - enemy can intercept)
		upScore := greedy.ScoreMove(state, board.MoveUp)       // Continue up (toward enemy path)
		downScore := greedy.ScoreMove(state, board.MoveDown)   // Back down (into our body)

		t.Logf("Scores:")
		t.Logf("  Left (wall U-turn): %.2f", leftScore)
		t.Logf("  Right (away from wall): %.2f", rightScore)
		t.Logf("  Up (toward enemy): %.2f", upScore)
		t.Logf("  Down (into body): %.2f", downScore)

		bestMove := greedy.FindBestMove(state)
		t.Logf("Best move chosen: %s", bestMove)

		// The critical check: when near wall with enemy approaching head-on,
		// turning into the wall should be scored favorably because it allows a U-turn
		// Currently, the snake might prefer turning right (away from wall) which leads to interception

		// Document current behavior
		if bestMove == board.MoveRight {
			t.Logf("WARNING: Snake chose to turn right (away from wall)")
			t.Logf("This could lead to enemy interception as described in the issue")
		} else if bestMove == board.MoveLeft {
			t.Logf("✓ Snake correctly chose to turn left into wall for defensive U-turn")
		}
	})

	t.Run("Exact issue scenario - 1 tile from wall, enemy 2 tiles ahead", func(t *testing.T) {
		// Most precise recreation of the issue:
		// Us: X=1, Y=5, moving up
		// Enemy: X=3, Y=7, moving down (about to be at Y=6)
		// Distance: 2 tiles vertically, 2 tiles horizontally
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
							{X: 1, Y: 4},
							{X: 1, Y: 3},
							{X: 2, Y: 3},
							{X: 3, Y: 3},
						},
						Head: board.Coord{X: 1, Y: 5},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 6, // Slightly larger - head-on is bad for us
						Body: []board.Coord{
							{X: 3, Y: 7}, // 2 tiles ahead (vertically)
							{X: 3, Y: 8},
							{X: 4, Y: 8},
							{X: 5, Y: 8},
							{X: 6, Y: 8},
							{X: 7, Y: 8},
						},
						Head: board.Coord{X: 3, Y: 7},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 5,
				Body: []board.Coord{
					{X: 1, Y: 5},
					{X: 1, Y: 4},
					{X: 1, Y: 3},
					{X: 2, Y: 3},
					{X: 3, Y: 3},
				},
				Head: board.Coord{X: 1, Y: 5},
			},
		}

		greedy := NewGreedySearch()

		leftScore := greedy.ScoreMove(state, board.MoveLeft)
		rightScore := greedy.ScoreMove(state, board.MoveRight)
		upScore := greedy.ScoreMove(state, board.MoveUp)
		downScore := greedy.ScoreMove(state, board.MoveDown)

		t.Logf("Issue scenario scores:")
		t.Logf("  Left (wall): %.2f", leftScore)
		t.Logf("  Right (away): %.2f", rightScore)
		t.Logf("  Up (continue): %.2f", upScore)
		t.Logf("  Down: %.2f", downScore)

		bestMove := greedy.FindBestMove(state)
		t.Logf("Chosen move: %s", bestMove)

		// This is the critical test case
		// The snake should recognize the danger of turning right
		// and prefer the wall turn (left) even though it seems constraining
		if bestMove == board.MoveRight && rightScore > leftScore {
			t.Errorf("BUG REPRODUCED: Snake prefers turning right (%.2f) over left wall turn (%.2f)",
				rightScore, leftScore)
			t.Errorf("This is the scenario from the issue - turning right leads to interception!")
		}
	})

	t.Run("Wall emergency - both same size, very close proximity", func(t *testing.T) {
		// Even tighter scenario: snakes are very close
		// Us: X=1, Y=6, moving up
		// Enemy: X=2, Y=8, moving down (1 tile diagonal distance)
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
							{X: 1, Y: 6}, // Near wall
							{X: 1, Y: 5},
							{X: 1, Y: 4},
							{X: 1, Y: 3},
							{X: 1, Y: 2},
						},
						Head: board.Coord{X: 1, Y: 6},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 5, // Same size
						Body: []board.Coord{
							{X: 2, Y: 8}, // Very close
							{X: 2, Y: 9},
							{X: 3, Y: 9},
							{X: 4, Y: 9},
							{X: 5, Y: 9},
						},
						Head: board.Coord{X: 2, Y: 8},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 5,
				Body: []board.Coord{
					{X: 1, Y: 6},
					{X: 1, Y: 5},
					{X: 1, Y: 4},
					{X: 1, Y: 3},
					{X: 1, Y: 2},
				},
				Head: board.Coord{X: 1, Y: 6},
			},
		}

		greedy := NewGreedySearch()

		leftScore := greedy.ScoreMove(state, board.MoveLeft)
		rightScore := greedy.ScoreMove(state, board.MoveRight)
		upScore := greedy.ScoreMove(state, board.MoveUp)
		downScore := greedy.ScoreMove(state, board.MoveDown)

		t.Logf("Close proximity scores:")
		t.Logf("  Left (wall): %.2f", leftScore)
		t.Logf("  Right: %.2f", rightScore)
		t.Logf("  Up: %.2f", upScore)
		t.Logf("  Down: %.2f", downScore)

		bestMove := greedy.FindBestMove(state)
		t.Logf("Chosen: %s", bestMove)

		// When enemy is this close and same size, we must be very defensive
		// The wall provides a constraint but allows a U-turn escape
	})
}
