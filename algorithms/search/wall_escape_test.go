package search

import (
	"testing"

	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

// TestWallSideHeadOnCollisionEscape validates the wall-side escape logic
// When snake is 1 tile from wall and enemy approaches head-on,
// snake should turn toward wall (safer) rather than away (guaranteed collision)
func TestWallSideHeadOnCollisionEscape(t *testing.T) {
	t.Run("Near left wall - perpendicular blocked - must turn toward wall", func(t *testing.T) {
		// Our snake is at X=1 (one tile from left wall at X=0)
		// Enemy is at X=3, approaching head-on
		// Perpendicular directions (up/down) are blocked by our body
		// Only choices: left (toward wall, safe) or right (toward enemy, fatal)
		state := &board.GameState{
			Turn: 50,
			Board: board.Board{
				Width:  11,
				Height: 11,
				Snakes: []board.Snake{
					{
						ID:     "me",
						Health: 70,
						Length: 8,
						Body: []board.Coord{
							{X: 1, Y: 5}, // Head - 1 tile from left wall
							{X: 1, Y: 4}, // Body blocks down
							{X: 1, Y: 3},
							{X: 2, Y: 3},
							{X: 2, Y: 4},
							{X: 2, Y: 5},
							{X: 1, Y: 6}, // Body blocks up
							{X: 1, Y: 7},
						},
						Head: board.Coord{X: 1, Y: 5},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 5,
						Body: []board.Coord{
							{X: 3, Y: 5}, // Head - 2 tiles away, head-on
							{X: 4, Y: 5},
							{X: 5, Y: 5},
							{X: 6, Y: 5},
							{X: 7, Y: 5},
						},
						Head: board.Coord{X: 3, Y: 5},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 8,
				Body: []board.Coord{
					{X: 1, Y: 5},
					{X: 1, Y: 4},
					{X: 1, Y: 3},
					{X: 2, Y: 3},
					{X: 2, Y: 4},
					{X: 2, Y: 5},
					{X: 1, Y: 6},
					{X: 1, Y: 7},
				},
				Head: board.Coord{X: 1, Y: 5},
			},
		}

		greedy := NewGreedySearch()

		// Score all moves
		leftScore := greedy.ScoreMove(state, board.MoveLeft)   // Toward wall (only safe option)
		rightScore := greedy.ScoreMove(state, board.MoveRight) // Head-on collision (fatal)
		upScore := greedy.ScoreMove(state, board.MoveUp)       // Blocked by body
		downScore := greedy.ScoreMove(state, board.MoveDown)   // Blocked by body

		t.Logf("Move scores - Left (wall): %.2f, Right (enemy): %.2f, Up: %.2f, Down: %.2f",
			leftScore, rightScore, upScore, downScore)

		bestMove := greedy.FindBestMove(state)

		// Right should have heavy penalty due to head-on collision risk
		if rightScore > leftScore {
			t.Errorf("FAILED: Moving toward enemy (right) scored higher (%.2f) than toward wall (left, %.2f)",
				rightScore, leftScore)
		}

		// Best move MUST be left (toward wall) since it's the only valid option
		if bestMove != board.MoveLeft {
			t.Errorf("FAILED: Snake chose to move %s instead of left toward wall!", bestMove)
		} else {
			t.Logf("✓ Best move: %s (correctly turns toward wall)", bestMove)
		}
	})

	t.Run("Near left wall - enemy head-on from right - turn left toward wall", func(t *testing.T) {
		// Our snake is at X=1 (one tile from left wall at X=0)
		// Enemy is at X=3, approaching head-on
		// If we turn right, we collide head-on with enemy
		// If we turn left, we go toward wall but survive
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
							{X: 1, Y: 5}, // Head - 1 tile from left wall
							{X: 2, Y: 5},
							{X: 3, Y: 5},
							{X: 4, Y: 5},
							{X: 5, Y: 5},
						},
						Head: board.Coord{X: 1, Y: 5},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 5,
						Body: []board.Coord{
							{X: 3, Y: 5}, // Head - 2 tiles away, head-on
							{X: 4, Y: 5},
							{X: 5, Y: 5},
							{X: 6, Y: 5},
							{X: 7, Y: 5},
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
					{X: 2, Y: 5},
					{X: 3, Y: 5},
					{X: 4, Y: 5},
					{X: 5, Y: 5},
				},
				Head: board.Coord{X: 1, Y: 5},
			},
		}

		greedy := NewGreedySearch()

		// Score all moves
		leftScore := greedy.ScoreMove(state, board.MoveLeft)   // Toward wall (safe)
		rightScore := greedy.ScoreMove(state, board.MoveRight) // Head-on collision (fatal)
		upScore := greedy.ScoreMove(state, board.MoveUp)       // Perpendicular
		downScore := greedy.ScoreMove(state, board.MoveDown)   // Perpendicular

		t.Logf("Move scores - Left (wall): %.2f, Right (enemy): %.2f, Up: %.2f, Down: %.2f",
			leftScore, rightScore, upScore, downScore)

		bestMove := greedy.FindBestMove(state)

		// Right should have heavy penalty due to head-on collision risk
		if rightScore > leftScore {
			t.Errorf("FAILED: Moving toward enemy (right) scored higher (%.2f) than toward wall (left, %.2f)",
				rightScore, leftScore)
		}

		// Best move should NOT be toward the enemy
		if bestMove == board.MoveRight {
			t.Errorf("FAILED: Snake chose to move right toward enemy - guaranteed head-on collision!")
		}

		// Perpendicular moves should be best, but if blocked, turning toward wall should be preferred over enemy
		t.Logf("✓ Best move: %s (correctly avoids head-on collision)", bestMove)
	})

	t.Run("Near right wall - enemy head-on from left - turn right toward wall", func(t *testing.T) {
		// Our snake is at X=9 (one tile from right wall at X=10)
		// Enemy is at X=7, approaching head-on
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
							{X: 9, Y: 5}, // Head - 1 tile from right wall
							{X: 8, Y: 5},
							{X: 7, Y: 5},
							{X: 6, Y: 5},
							{X: 5, Y: 5},
						},
						Head: board.Coord{X: 9, Y: 5},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 5,
						Body: []board.Coord{
							{X: 7, Y: 5}, // Head - 2 tiles away, head-on
							{X: 6, Y: 5},
							{X: 5, Y: 5},
							{X: 4, Y: 5},
							{X: 3, Y: 5},
						},
						Head: board.Coord{X: 7, Y: 5},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 5,
				Body: []board.Coord{
					{X: 9, Y: 5},
					{X: 8, Y: 5},
					{X: 7, Y: 5},
					{X: 6, Y: 5},
					{X: 5, Y: 5},
				},
				Head: board.Coord{X: 9, Y: 5},
			},
		}

		greedy := NewGreedySearch()

		rightScore := greedy.ScoreMove(state, board.MoveRight) // Toward wall (safe)
		leftScore := greedy.ScoreMove(state, board.MoveLeft)   // Head-on collision (fatal)

		t.Logf("Move scores - Right (wall): %.2f, Left (enemy): %.2f", rightScore, leftScore)

		bestMove := greedy.FindBestMove(state)

		// Left should have heavy penalty due to head-on collision risk
		if leftScore > rightScore {
			t.Errorf("FAILED: Moving toward enemy (left) scored higher (%.2f) than toward wall (right, %.2f)",
				leftScore, rightScore)
		}

		// Best move should NOT be toward the enemy
		if bestMove == board.MoveLeft {
			t.Errorf("FAILED: Snake chose to move left toward enemy - guaranteed head-on collision!")
		}

		t.Logf("✓ Best move: %s (correctly avoids head-on collision)", bestMove)
	})

	t.Run("Near top wall - enemy head-on from below - turn up toward wall", func(t *testing.T) {
		// Our snake is at Y=9 (one tile from top wall at Y=10)
		// Enemy is at Y=7, approaching head-on from below
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
							{X: 5, Y: 9}, // Head - 1 tile from top wall
							{X: 5, Y: 8},
							{X: 5, Y: 7},
							{X: 5, Y: 6},
							{X: 5, Y: 5},
						},
						Head: board.Coord{X: 5, Y: 9},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 5,
						Body: []board.Coord{
							{X: 5, Y: 7}, // Head - 2 tiles away, head-on
							{X: 5, Y: 6},
							{X: 5, Y: 5},
							{X: 5, Y: 4},
							{X: 5, Y: 3},
						},
						Head: board.Coord{X: 5, Y: 7},
					},
				},
			},
			You: board.Snake{
				ID:     "me",
				Health: 70,
				Length: 5,
				Body: []board.Coord{
					{X: 5, Y: 9},
					{X: 5, Y: 8},
					{X: 5, Y: 7},
					{X: 5, Y: 6},
					{X: 5, Y: 5},
				},
				Head: board.Coord{X: 5, Y: 9},
			},
		}

		greedy := NewGreedySearch()

		upScore := greedy.ScoreMove(state, board.MoveUp)     // Toward wall (safe)
		downScore := greedy.ScoreMove(state, board.MoveDown) // Head-on collision (fatal)

		t.Logf("Move scores - Up (wall): %.2f, Down (enemy): %.2f", upScore, downScore)

		bestMove := greedy.FindBestMove(state)

		// Down should have heavy penalty due to head-on collision risk
		if downScore > upScore {
			t.Errorf("FAILED: Moving toward enemy (down) scored higher (%.2f) than toward wall (up, %.2f)",
				downScore, upScore)
		}

		// Best move should NOT be toward the enemy
		if bestMove == board.MoveDown {
			t.Errorf("FAILED: Snake chose to move down toward enemy - guaranteed head-on collision!")
		}

		t.Logf("✓ Best move: %s (correctly avoids head-on collision)", bestMove)
	})

	t.Run("Critical scenario - enemy blocks escape away from wall", func(t *testing.T) {
		// Our snake moving left toward wall (at X=1, came from X=2)
		// Enemy approaching head-on from left (at X=-1 conceptually, about to be at X=0)
		// If we turn right (away from wall), we turn INTO the enemy path
		// If we turn up/down (perpendicular), safer but may be limited space
		// But most critically: turning away from wall should be recognized as dangerous
		//
		// More realistic: Snake at X=1, moving left, enemy at X=0 moving right
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
							{X: 1, Y: 5}, // Head - 1 tile from left wall
							{X: 2, Y: 5}, // Neck - we were moving left
							{X: 3, Y: 5},
							{X: 4, Y: 5},
							{X: 5, Y: 5},
						},
						Head: board.Coord{X: 1, Y: 5},
					},
					{
						ID:     "enemy",
						Health: 70,
						Length: 6, // Slightly larger - head-on is fatal for us
						Body: []board.Coord{
							{X: 2, Y: 6}, // Enemy positioned so turning right puts us in danger
							{X: 3, Y: 6},
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

		leftScore := greedy.ScoreMove(state, board.MoveLeft)   // Toward wall
		rightScore := greedy.ScoreMove(state, board.MoveRight) // Away from wall but back toward body
		upScore := greedy.ScoreMove(state, board.MoveUp)       // Perpendicular - toward enemy
		downScore := greedy.ScoreMove(state, board.MoveDown)   // Perpendicular - away from enemy

		t.Logf("Move scores - Left (wall): %.2f, Right (away, into body): %.2f, Up (toward enemy): %.2f, Down: %.2f",
			leftScore, rightScore, upScore, downScore)

		bestMove := greedy.FindBestMove(state)
		t.Logf("✓ Best move: %s", bestMove)

		// The critical insight: when near wall with enemy nearby,
		// the snake must evaluate collision danger properly
		// This test documents current behavior for the scenario
	})
}
