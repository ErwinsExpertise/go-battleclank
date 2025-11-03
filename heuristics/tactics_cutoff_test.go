package heuristics

import (
	"testing"

	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

// TestDetectCutoffKill_BasicLeftWallScenario tests the basic cutoff kill on the left wall
func TestDetectCutoffKill_BasicLeftWallScenario(t *testing.T) {
	// Setup: Both snakes moving up, enemy on left wall (x=0), we're at x=1
	// Our head is at (1, 5), enemy head at (0, 3) - we're 2 ahead
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food:   []board.Coord{},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Health: 80,
					Body: []board.Coord{
						{X: 1, Y: 5}, // Head - moving up
						{X: 1, Y: 4}, // Neck
						{X: 1, Y: 3}, // Body extending past enemy head
						{X: 1, Y: 2}, // Body extending past enemy head
						{X: 1, Y: 1},
					},
					Head:   board.Coord{X: 1, Y: 5},
					Length: 5,
				},
				{
					ID:     "enemy",
					Health: 80,
					Body: []board.Coord{
						{X: 0, Y: 3}, // Head on left wall - moving up
						{X: 0, Y: 2}, // Neck
						{X: 0, Y: 1},
					},
					Head:   board.Coord{X: 0, Y: 3},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Health: 80,
			Body: []board.Coord{
				{X: 1, Y: 5},
				{X: 1, Y: 4},
				{X: 1, Y: 3},
				{X: 1, Y: 2},
				{X: 1, Y: 1},
			},
			Head:   board.Coord{X: 1, Y: 5},
			Length: 5,
		},
	}

	cutoffMove := DetectCutoffKill(state)

	if cutoffMove != board.MoveLeft {
		t.Errorf("Expected cutoff kill move to be 'left', got '%s'", cutoffMove)
	}
	t.Logf("✓ Correctly detected cutoff kill opportunity on left wall")
}

// TestDetectCutoffKill_RightWallScenario tests cutoff kill on the right wall
func TestDetectCutoffKill_RightWallScenario(t *testing.T) {
	// Enemy on right wall (x=10), we're at x=9, both moving down
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food:   []board.Coord{},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Health: 80,
					Body: []board.Coord{
						{X: 9, Y: 5}, // Head - moving down
						{X: 9, Y: 6}, // Neck
						{X: 9, Y: 7}, // Body extending past enemy head
						{X: 9, Y: 8}, // Body extending past enemy head
						{X: 9, Y: 9},
					},
					Head:   board.Coord{X: 9, Y: 5},
					Length: 5,
				},
				{
					ID:     "enemy",
					Health: 80,
					Body: []board.Coord{
						{X: 10, Y: 7}, // Head on right wall - moving down
						{X: 10, Y: 8}, // Neck
						{X: 10, Y: 9},
					},
					Head:   board.Coord{X: 10, Y: 7},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Health: 80,
			Body: []board.Coord{
				{X: 9, Y: 5},
				{X: 9, Y: 6},
				{X: 9, Y: 7},
				{X: 9, Y: 8},
				{X: 9, Y: 9},
			},
			Head:   board.Coord{X: 9, Y: 5},
			Length: 5,
		},
	}

	cutoffMove := DetectCutoffKill(state)

	if cutoffMove != board.MoveRight {
		t.Errorf("Expected cutoff kill move to be 'right', got '%s'", cutoffMove)
	}
	t.Logf("✓ Correctly detected cutoff kill opportunity on right wall")
}

// TestDetectCutoffKill_TopWallScenario tests cutoff kill on the top wall
func TestDetectCutoffKill_TopWallScenario(t *testing.T) {
	// Enemy on top wall (y=10), we're at y=9, both moving right
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food:   []board.Coord{},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Health: 80,
					Body: []board.Coord{
						{X: 5, Y: 9}, // Head - moving right
						{X: 4, Y: 9}, // Neck
						{X: 3, Y: 9}, // Body extending past enemy head
						{X: 2, Y: 9}, // Body extending past enemy head
						{X: 1, Y: 9},
					},
					Head:   board.Coord{X: 5, Y: 9},
					Length: 5,
				},
				{
					ID:     "enemy",
					Health: 80,
					Body: []board.Coord{
						{X: 3, Y: 10}, // Head on top wall - moving right
						{X: 2, Y: 10}, // Neck
						{X: 1, Y: 10},
					},
					Head:   board.Coord{X: 3, Y: 10},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Health: 80,
			Body: []board.Coord{
				{X: 5, Y: 9},
				{X: 4, Y: 9},
				{X: 3, Y: 9},
				{X: 2, Y: 9},
				{X: 1, Y: 9},
			},
			Head:   board.Coord{X: 5, Y: 9},
			Length: 5,
		},
	}

	cutoffMove := DetectCutoffKill(state)

	if cutoffMove != board.MoveUp {
		t.Errorf("Expected cutoff kill move to be 'up', got '%s'", cutoffMove)
	}
	t.Logf("✓ Correctly detected cutoff kill opportunity on top wall")
}

// TestDetectCutoffKill_BottomWallScenario tests cutoff kill on the bottom wall
func TestDetectCutoffKill_BottomWallScenario(t *testing.T) {
	// Enemy on bottom wall (y=0), we're at y=1, both moving left
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food:   []board.Coord{},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Health: 80,
					Body: []board.Coord{
						{X: 5, Y: 1}, // Head - moving left
						{X: 6, Y: 1}, // Neck
						{X: 7, Y: 1}, // Body extending past enemy head
						{X: 8, Y: 1}, // Body extending past enemy head
						{X: 9, Y: 1},
					},
					Head:   board.Coord{X: 5, Y: 1},
					Length: 5,
				},
				{
					ID:     "enemy",
					Health: 80,
					Body: []board.Coord{
						{X: 7, Y: 0}, // Head on bottom wall - moving left
						{X: 8, Y: 0}, // Neck
						{X: 9, Y: 0},
					},
					Head:   board.Coord{X: 7, Y: 0},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Health: 80,
			Body: []board.Coord{
				{X: 5, Y: 1},
				{X: 6, Y: 1},
				{X: 7, Y: 1},
				{X: 8, Y: 1},
				{X: 9, Y: 1},
			},
			Head:   board.Coord{X: 5, Y: 1},
			Length: 5,
		},
	}

	cutoffMove := DetectCutoffKill(state)

	if cutoffMove != board.MoveDown {
		t.Errorf("Expected cutoff kill move to be 'down', got '%s'", cutoffMove)
	}
	t.Logf("✓ Correctly detected cutoff kill opportunity on bottom wall")
}

// TestDetectCutoffKill_NotEnoughAdvance tests that cutoff is NOT triggered when we're not far enough ahead
func TestDetectCutoffKill_NotEnoughAdvance(t *testing.T) {
	// Same setup but we're only 1 tile ahead (need 2)
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food:   []board.Coord{},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Health: 80,
					Body: []board.Coord{
						{X: 1, Y: 4}, // Head - only 1 ahead
						{X: 1, Y: 3}, // Neck
						{X: 1, Y: 2},
					},
					Head:   board.Coord{X: 1, Y: 4},
					Length: 3,
				},
				{
					ID:     "enemy",
					Health: 80,
					Body: []board.Coord{
						{X: 0, Y: 3}, // Head on left wall
						{X: 0, Y: 2}, // Neck
						{X: 0, Y: 1},
					},
					Head:   board.Coord{X: 0, Y: 3},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Health: 80,
			Body: []board.Coord{
				{X: 1, Y: 4},
				{X: 1, Y: 3},
				{X: 1, Y: 2},
			},
			Head:   board.Coord{X: 1, Y: 4},
			Length: 3,
		},
	}

	cutoffMove := DetectCutoffKill(state)

	if cutoffMove != "" {
		t.Errorf("Expected no cutoff kill (not enough advance), but got '%s'", cutoffMove)
	}
	t.Logf("✓ Correctly skipped cutoff when not far enough ahead")
}

// TestDetectCutoffKill_NotEnoughBodyExtension tests that cutoff is NOT triggered without enough body extension
func TestDetectCutoffKill_NotEnoughBodyExtension(t *testing.T) {
	// We're ahead but body doesn't extend at least 2 tiles past enemy head
	// Our head is only 1 tile past enemy head (not 2+)
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food:   []board.Coord{},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Health: 80,
					Body: []board.Coord{
						{X: 1, Y: 4}, // Head - only 1 tile past enemy at y=3
						{X: 1, Y: 3}, // At same level as enemy
						{X: 1, Y: 2}, // Behind enemy
					},
					Head:   board.Coord{X: 1, Y: 4},
					Length: 3,
				},
				{
					ID:     "enemy",
					Health: 80,
					Body: []board.Coord{
						{X: 0, Y: 3}, // Head on left wall
						{X: 0, Y: 2}, // Neck
						{X: 0, Y: 1},
					},
					Head:   board.Coord{X: 0, Y: 3},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Health: 80,
			Body: []board.Coord{
				{X: 1, Y: 4},
				{X: 1, Y: 3},
				{X: 1, Y: 2},
			},
			Head:   board.Coord{X: 1, Y: 4},
			Length: 3,
		},
	}

	cutoffMove := DetectCutoffKill(state)

	if cutoffMove != "" {
		t.Errorf("Expected no cutoff kill (not enough body extension), but got '%s'", cutoffMove)
	}
	t.Logf("✓ Correctly skipped cutoff when body doesn't extend far enough")
}

// TestDetectCutoffKill_DifferentDirections tests that cutoff is NOT triggered when snakes move in different directions
func TestDetectCutoffKill_DifferentDirections(t *testing.T) {
	// Enemy moving up, we're moving right - no cutoff possible
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food:   []board.Coord{},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Health: 80,
					Body: []board.Coord{
						{X: 5, Y: 5}, // Head - moving right
						{X: 4, Y: 5}, // Neck
						{X: 3, Y: 5},
					},
					Head:   board.Coord{X: 5, Y: 5},
					Length: 3,
				},
				{
					ID:     "enemy",
					Health: 80,
					Body: []board.Coord{
						{X: 0, Y: 3}, // Head - moving up (different direction)
						{X: 0, Y: 2}, // Neck
						{X: 0, Y: 1},
					},
					Head:   board.Coord{X: 0, Y: 3},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Health: 80,
			Body: []board.Coord{
				{X: 5, Y: 5},
				{X: 4, Y: 5},
				{X: 3, Y: 5},
			},
			Head:   board.Coord{X: 5, Y: 5},
			Length: 3,
		},
	}

	cutoffMove := DetectCutoffKill(state)

	if cutoffMove != "" {
		t.Errorf("Expected no cutoff kill (different directions), but got '%s'", cutoffMove)
	}
	t.Logf("✓ Correctly skipped cutoff when snakes move in different directions")
}

// TestDetectCutoffKill_EnemyInCorner tests that cutoff is NOT triggered when enemy is in a corner
func TestDetectCutoffKill_EnemyInCorner(t *testing.T) {
	// Enemy in corner (0, 0) - should skip to avoid self-collision risk
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food:   []board.Coord{},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Health: 80,
					Body: []board.Coord{
						{X: 1, Y: 5}, // Head - moving up
						{X: 1, Y: 4}, // Neck
						{X: 1, Y: 3},
						{X: 1, Y: 2},
						{X: 1, Y: 1},
					},
					Head:   board.Coord{X: 1, Y: 5},
					Length: 5,
				},
				{
					ID:     "enemy",
					Health: 80,
					Body: []board.Coord{
						{X: 0, Y: 0}, // Head in corner
						{X: 0, Y: 1}, // Neck
						{X: 0, Y: 2},
					},
					Head:   board.Coord{X: 0, Y: 0},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Health: 80,
			Body: []board.Coord{
				{X: 1, Y: 5},
				{X: 1, Y: 4},
				{X: 1, Y: 3},
				{X: 1, Y: 2},
				{X: 1, Y: 1},
			},
			Head:   board.Coord{X: 1, Y: 5},
			Length: 5,
		},
	}

	cutoffMove := DetectCutoffKill(state)

	if cutoffMove != "" {
		t.Errorf("Expected no cutoff kill (enemy in corner), but got '%s'", cutoffMove)
	}
	t.Logf("✓ Correctly skipped cutoff when enemy is in corner")
}

// TestDetectCutoffKill_WeAreInCorner tests that cutoff is NOT triggered when we're in a corner
func TestDetectCutoffKill_WeAreInCorner(t *testing.T) {
	// We're in corner - should skip to avoid self-collision risk
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food:   []board.Coord{},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Health: 80,
					Body: []board.Coord{
						{X: 0, Y: 0}, // Head in corner - moving left (from 1,0 to 0,0)
						{X: 1, Y: 0}, // Neck
						{X: 2, Y: 0},
					},
					Head:   board.Coord{X: 0, Y: 0},
					Length: 3,
				},
				{
					ID:     "enemy",
					Health: 80,
					Body: []board.Coord{
						{X: 0, Y: 1}, // Head on wall - moving left (from 1,1 to 0,1)
						{X: 1, Y: 1}, // Neck
						{X: 2, Y: 1},
					},
					Head:   board.Coord{X: 0, Y: 1},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Health: 80,
			Body: []board.Coord{
				{X: 0, Y: 0},
				{X: 1, Y: 0},
				{X: 2, Y: 0},
			},
			Head:   board.Coord{X: 0, Y: 0},
			Length: 3,
		},
	}

	cutoffMove := DetectCutoffKill(state)

	if cutoffMove != "" {
		t.Errorf("Expected no cutoff kill (we're in corner), but got '%s'", cutoffMove)
	}
	t.Logf("✓ Correctly skipped cutoff when we're in corner")
}

// TestDetectCutoffKill_WeAreNotOneInward tests that cutoff is NOT triggered when we're not exactly one lane inward
func TestDetectCutoffKill_WeAreNotOneInward(t *testing.T) {
	// Enemy on left wall (x=0), but we're at x=2 (should be x=1)
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food:   []board.Coord{},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Health: 80,
					Body: []board.Coord{
						{X: 2, Y: 5}, // Head at x=2 (too far inward)
						{X: 2, Y: 4}, // Neck
						{X: 2, Y: 3},
						{X: 2, Y: 2},
					},
					Head:   board.Coord{X: 2, Y: 5},
					Length: 4,
				},
				{
					ID:     "enemy",
					Health: 80,
					Body: []board.Coord{
						{X: 0, Y: 3}, // Head on left wall
						{X: 0, Y: 2}, // Neck
						{X: 0, Y: 1},
					},
					Head:   board.Coord{X: 0, Y: 3},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Health: 80,
			Body: []board.Coord{
				{X: 2, Y: 5},
				{X: 2, Y: 4},
				{X: 2, Y: 3},
				{X: 2, Y: 2},
			},
			Head:   board.Coord{X: 2, Y: 5},
			Length: 4,
		},
	}

	cutoffMove := DetectCutoffKill(state)

	if cutoffMove != "" {
		t.Errorf("Expected no cutoff kill (not one lane inward), but got '%s'", cutoffMove)
	}
	t.Logf("✓ Correctly skipped cutoff when not exactly one lane inward")
}

// TestDetectCutoffKill_EnemyNotOnWall tests that cutoff is NOT triggered when enemy is not on wall
func TestDetectCutoffKill_EnemyNotOnWall(t *testing.T) {
	// Enemy at x=1 (not on wall), we're at x=2
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Width:  11,
			Height: 11,
			Food:   []board.Coord{},
			Snakes: []board.Snake{
				{
					ID:     "us",
					Health: 80,
					Body: []board.Coord{
						{X: 2, Y: 5}, // Head - moving up
						{X: 2, Y: 4}, // Neck
						{X: 2, Y: 3},
						{X: 2, Y: 2},
					},
					Head:   board.Coord{X: 2, Y: 5},
					Length: 4,
				},
				{
					ID:     "enemy",
					Health: 80,
					Body: []board.Coord{
						{X: 1, Y: 3}, // Head NOT on wall
						{X: 1, Y: 2}, // Neck
						{X: 1, Y: 1},
					},
					Head:   board.Coord{X: 1, Y: 3},
					Length: 3,
				},
			},
		},
		You: board.Snake{
			ID:     "us",
			Health: 80,
			Body: []board.Coord{
				{X: 2, Y: 5},
				{X: 2, Y: 4},
				{X: 2, Y: 3},
				{X: 2, Y: 2},
			},
			Head:   board.Coord{X: 2, Y: 5},
			Length: 4,
		},
	}

	cutoffMove := DetectCutoffKill(state)

	if cutoffMove != "" {
		t.Errorf("Expected no cutoff kill (enemy not on wall), but got '%s'", cutoffMove)
	}
	t.Logf("✓ Correctly skipped cutoff when enemy is not on wall")
}
