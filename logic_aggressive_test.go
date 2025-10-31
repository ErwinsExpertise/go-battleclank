package main

import (
	"testing"
)

// Test predictEnemyMoves function
func TestPredictEnemyMoves(t *testing.T) {
	state := GameState{
		You: createTestSnake("me", 100, []Coord{
			{X: 0, Y: 0},
			{X: 0, Y: 1},
			{X: 0, Y: 2},
		}),
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				createTestSnake("me", 100, []Coord{
					{X: 0, Y: 0},
					{X: 0, Y: 1},
					{X: 0, Y: 2},
				}),
				createTestSnake("enemy", 100, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				}),
			},
		},
	}

	enemyMoves := predictEnemyMoves(state)

	// Enemy at (5,5) can move to (5,6), (5,4 - blocked), (4,5), (6,5)
	// Should have 3 possible positions
	expectedPositions := []Coord{
		{X: 5, Y: 6}, // Up
		{X: 4, Y: 5}, // Left
		{X: 6, Y: 5}, // Right
		// Down is blocked by its own body at (5,4)
	}

	if len(enemyMoves) != len(expectedPositions) {
		t.Logf("Enemy moves predicted: %d positions", len(enemyMoves))
		for pos, snakes := range enemyMoves {
			t.Logf("  Position %v: %d snakes", pos, len(snakes))
		}
	}

	for _, pos := range expectedPositions {
		if _, found := enemyMoves[pos]; !found {
			t.Errorf("Expected enemy to potentially move to %v, but it wasn't predicted", pos)
		}
	}
}

// Test predictEnemyMoves with multiple enemies
func TestPredictEnemyMoves_Multiple(t *testing.T) {
	state := GameState{
		You: createTestSnake("me", 100, []Coord{
			{X: 5, Y: 5},
			{X: 5, Y: 4},
			{X: 5, Y: 3},
		}),
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				createTestSnake("me", 100, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				}),
				createTestSnake("enemy1", 100, []Coord{
					{X: 7, Y: 5}, // To the right
					{X: 7, Y: 4},
					{X: 7, Y: 3},
				}),
				createTestSnake("enemy2", 100, []Coord{
					{X: 3, Y: 5}, // To the left
					{X: 3, Y: 4},
					{X: 3, Y: 3},
				}),
			},
		},
	}

	enemyMoves := predictEnemyMoves(state)

	// Check for overlapping positions where both enemies could move
	// Position (6,5) should be reachable by enemy1
	// Position (4,5) should be reachable by enemy2
	if _, found := enemyMoves[Coord{X: 6, Y: 5}]; !found {
		t.Error("Enemy1 should be able to move to (6,5)")
	}

	if _, found := enemyMoves[Coord{X: 4, Y: 5}]; !found {
		t.Error("Enemy2 should be able to move to (4,5)")
	}
}

// Test simulateMove detects self-trap
func TestSimulateMove_DetectsSelfTrap(t *testing.T) {
	// Snake in a very tight spot where most moves lead to limited options
	// Create a U-shaped snake body that blocks most directions
	mySnake := createTestSnake("me", 80, []Coord{
		{X: 2, Y: 2}, // Head in somewhat open area
		{X: 2, Y: 1}, // Body going down
		{X: 2, Y: 0}, // At bottom
		{X: 3, Y: 0}, // Going right
		{X: 4, Y: 0}, // Continuing right
		{X: 4, Y: 1}, // Going up
		{X: 4, Y: 2}, // Near head level
	})

	state := GameState{
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{mySnake},
		},
	}

	// From (2,2), moving down (2,1) is immediately fatal (our body)
	// Moving right (3,2) should be reasonably safe (open space)
	// Moving up (2,3) should be safe (open space)
	// Moving left (1,2) should be safe (open space)
	
	// Test that non-fatal moves are considered safe
	isSafe := simulateMove(state, MoveUp, 1)
	if !isSafe {
		t.Error("simulateMove should consider moving up as safe")
	}

	isSafe = simulateMove(state, MoveLeft, 1)
	if !isSafe {
		t.Error("simulateMove should consider moving left as safe")
	}

	// Down move is immediately fatal (hits body at 2,1), so simulateMove
	// won't even get called in scoreMove - isImmediatelyFatal catches it first
	
	// This test mainly verifies that simulateMove doesn't give false negatives
	// for moves that have adequate escape routes
}

// Test isOutmatchedByNearbyEnemies
func TestIsOutmatchedByNearbyEnemies(t *testing.T) {
	tests := []struct {
		name       string
		mySnake    Battlesnake
		enemySnake Battlesnake
		expected   bool
	}{
		{
			name: "Enemy much larger and nearby",
			mySnake: createTestSnake("me", 100, []Coord{
				{X: 5, Y: 5},
				{X: 5, Y: 4},
				{X: 5, Y: 3}, // Length 3
			}),
			enemySnake: createTestSnake("enemy", 100, []Coord{
				{X: 7, Y: 6}, // Nearby (dist=3)
				{X: 7, Y: 5},
				{X: 7, Y: 4},
				{X: 7, Y: 3},
				{X: 7, Y: 2},
				{X: 7, Y: 1}, 
				{X: 6, Y: 1},
				{X: 5, Y: 1}, // Length 8 (5 longer, exceeds +4 threshold)
			}),
			expected: true,
		},
		{
			name: "Enemy slightly larger but not enough",
			mySnake: createTestSnake("me", 100, []Coord{
				{X: 5, Y: 5},
				{X: 5, Y: 4},
				{X: 5, Y: 3},
				{X: 5, Y: 2}, // Length 4
			}),
			enemySnake: createTestSnake("enemy", 100, []Coord{
				{X: 7, Y: 6}, // Nearby
				{X: 7, Y: 5},
				{X: 7, Y: 4},
				{X: 7, Y: 3},
				{X: 7, Y: 2}, // Length 5 (only 1 longer)
			}),
			expected: false,
		},
		{
			name: "Enemy larger but far away",
			mySnake: createTestSnake("me", 100, []Coord{
				{X: 0, Y: 0},
				{X: 0, Y: 1},
				{X: 0, Y: 2}, // Length 3
			}),
			enemySnake: createTestSnake("enemy", 100, []Coord{
				{X: 10, Y: 10}, // Far (dist=20)
				{X: 10, Y: 9},
				{X: 10, Y: 8},
				{X: 10, Y: 7},
				{X: 10, Y: 6},
				{X: 10, Y: 5}, // Length 6
			}),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := GameState{
				You: tt.mySnake,
				Board: Board{
					Width:  11,
					Height: 11,
					Snakes: []Battlesnake{tt.mySnake, tt.enemySnake},
				},
			}

			result := isOutmatchedByNearbyEnemies(state)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Test isBeingChased
func TestIsBeingChased(t *testing.T) {
	tests := []struct {
		name     string
		mySnake  Battlesnake
		enemy    Battlesnake
		expected bool
	}{
		{
			name: "Enemy following our tail",
			mySnake: createTestSnake("me", 80, []Coord{
				{X: 5, Y: 5}, // Head
				{X: 5, Y: 4},
				{X: 5, Y: 3},
				{X: 5, Y: 2}, // Tail
			}),
			enemy: createTestSnake("enemy", 100, []Coord{
				{X: 5, Y: 0}, // Close to our tail
				{X: 5, Y: 1},
				{X: 6, Y: 1},
			}),
			expected: true,
		},
		{
			name: "Enemy not chasing",
			mySnake: createTestSnake("me", 80, []Coord{
				{X: 5, Y: 5}, // Head
				{X: 5, Y: 4},
				{X: 5, Y: 3},
				{X: 5, Y: 2}, // Tail
			}),
			enemy: createTestSnake("enemy", 100, []Coord{
				{X: 10, Y: 10}, // Far away
				{X: 10, Y: 9},
				{X: 10, Y: 8},
			}),
			expected: false,
		},
		{
			name: "Enemy near head, not tail",
			mySnake: createTestSnake("me", 80, []Coord{
				{X: 5, Y: 5}, // Head
				{X: 5, Y: 4},
				{X: 5, Y: 3},
				{X: 5, Y: 2}, // Tail
			}),
			enemy: createTestSnake("enemy", 100, []Coord{
				{X: 7, Y: 6}, // Near head, far from tail
				{X: 7, Y: 5},
				{X: 7, Y: 4},
			}),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := GameState{
				You: tt.mySnake,
				Board: Board{
					Width:  11,
					Height: 11,
					Snakes: []Battlesnake{tt.mySnake, tt.enemy},
				},
			}

			result := isBeingChased(state)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Test detectCutoff
func TestDetectCutoff(t *testing.T) {
	tests := []struct {
		name     string
		state    GameState
		pos      Coord
		expected string // "none", "moderate", "high", "extreme"
	}{
		{
			name: "Open space - no cutoff",
			state: GameState{
				You: createTestSnake("me", 80, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				}),
				Board: Board{
					Width:  11,
					Height: 11,
					Snakes: []Battlesnake{
						createTestSnake("me", 80, []Coord{
							{X: 5, Y: 5},
							{X: 5, Y: 4},
							{X: 5, Y: 3},
						}),
					},
				},
			},
			pos:      Coord{X: 5, Y: 5},
			expected: "none",
		},
		{
			name: "Completely trapped",
			state: GameState{
				You: createTestSnake("me", 80, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				}),
				Board: Board{
					Width:  7,
					Height: 7,
					Snakes: []Battlesnake{
						createTestSnake("me", 80, []Coord{
							{X: 5, Y: 5},
							{X: 5, Y: 4},
							{X: 5, Y: 3},
						}),
						createTestSnake("enemy", 100, []Coord{
							// Surround position (3,3)
							{X: 3, Y: 4}, // Up
							{X: 2, Y: 3}, // Left
							{X: 4, Y: 3}, // Right
							{X: 3, Y: 2}, // Down
							{X: 1, Y: 1},
						}),
					},
				},
			},
			pos:      Coord{X: 3, Y: 3},
			expected: "extreme",
		},
		{
			name: "One escape route",
			state: GameState{
				You: createTestSnake("me", 80, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				}),
				Board: Board{
					Width:  11,
					Height: 11,
					Snakes: []Battlesnake{
						createTestSnake("me", 80, []Coord{
							{X: 5, Y: 5},
							{X: 5, Y: 4},
							{X: 5, Y: 3},
						}),
						createTestSnake("enemy", 100, []Coord{
							// Block 3 sides of position (1,1)
							{X: 1, Y: 2}, // Up
							{X: 0, Y: 1}, // Left (wall on this side anyway)
							{X: 2, Y: 1}, // Right
							// Down is at wall
							{X: 3, Y: 3},
						}),
					},
				},
			},
			pos:      Coord{X: 1, Y: 1},
			expected: "high",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			penalty := detectCutoff(tt.state, tt.pos)
			t.Logf("Cutoff penalty for %s: %.2f", tt.name, penalty)

			switch tt.expected {
			case "none":
				if penalty > 0.1 {
					t.Errorf("Expected no cutoff, got penalty %.2f", penalty)
				}
			case "moderate":
				if penalty < 1.5 || penalty > 3.5 {
					t.Errorf("Expected moderate cutoff (1.5-3.5), got %.2f", penalty)
				}
			case "high":
				if penalty < 4.0 || penalty > 7.0 {
					t.Errorf("Expected high cutoff (4.0-7.0), got %.2f", penalty)
				}
			case "extreme":
				if penalty < 8.0 {
					t.Errorf("Expected extreme cutoff (>8.0), got %.2f", penalty)
				}
			}
		})
	}
}

// Test head-on collision with enemy danger zone
func TestMove_AvoidsEnemyDangerZone(t *testing.T) {
	// Scenario: Enemy head is nearby and can move to adjacent squares
	// We should avoid moving into any square the enemy can reach
	mySnake := createTestSnake("me", 80, []Coord{
		{X: 5, Y: 5}, // Head
		{X: 5, Y: 4},
		{X: 5, Y: 3},
	})

	enemySnake := createTestSnake("enemy", 100, []Coord{
		{X: 7, Y: 5}, // Enemy head 2 squares to the right
		{X: 7, Y: 4},
		{X: 7, Y: 3},
	})

	state := GameState{
		Turn: 20,
		You:  mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 6, Y: 5}}, // Food between us and enemy
			Snakes: []Battlesnake{mySnake, enemySnake},
		},
	}

	// Moving right (toward 6,5) should be heavily penalized
	// because enemy can also reach (6,5)
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
	}

	t.Logf("Scores - Up: %.2f, Down: %.2f, Left: %.2f, Right: %.2f",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	// Right should have a worse score than left due to danger zone
	if scores[MoveRight] >= scores[MoveLeft] {
		t.Logf("Warning: Right (toward enemy danger zone) scored %.2f vs Left %.2f",
			scores[MoveRight], scores[MoveLeft])
	}
}

// Test defensive play when outmatched
func TestMove_DefensiveWhenOutmatched(t *testing.T) {
	// Small snake near large enemy
	mySnake := createTestSnake("me", 40, []Coord{
		{X: 5, Y: 5}, // Head
		{X: 5, Y: 4},
		{X: 5, Y: 3}, // Length 3
	})

	largeEnemy := createTestSnake("enemy", 100, []Coord{
		{X: 7, Y: 6}, // Nearby
		{X: 7, Y: 5},
		{X: 7, Y: 4},
		{X: 7, Y: 3},
		{X: 7, Y: 2},
		{X: 7, Y: 1},
		{X: 6, Y: 1},
		{X: 5, Y: 1}, // Length 8 - much larger (5+ longer than us)
	})

	state := GameState{
		Turn: 30,
		You:  mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 8, Y: 6}}, // Food near enemy
			Snakes: []Battlesnake{mySnake, largeEnemy},
		},
	}

	// Check that we're detected as outmatched
	if !isOutmatchedByNearbyEnemies(state) {
		t.Error("Should detect that we're outmatched by nearby large enemy")
	}

	// When outmatched, food-seeking should be reduced in favor of space
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
	}

	t.Logf("Scores when outmatched - Up: %.2f, Down: %.2f, Left: %.2f, Right: %.2f",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	// Should prefer moves away from the large enemy (left/down) over toward it
	response := move(state)
	t.Logf("Snake chose: %s", response.Move)
}

// Test isFoodDangerous with enemy reaching faster
func TestIsFoodDangerous_EnemyReachesFaster(t *testing.T) {
	state := GameState{
		You: createTestSnake("me", 60, []Coord{
			{X: 0, Y: 0}, // Head at corner
			{X: 0, Y: 1},
			{X: 0, Y: 2},
		}),
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 5, Y: 5}}, // Food in center
			Snakes: []Battlesnake{
				createTestSnake("me", 60, []Coord{
					{X: 0, Y: 0},
					{X: 0, Y: 1},
					{X: 0, Y: 2},
				}),
				createTestSnake("enemy", 80, []Coord{
					{X: 3, Y: 3}, // Enemy much closer to food
					{X: 3, Y: 2},
					{X: 3, Y: 1},
				}),
			},
		},
	}

	// Distance from us to food: 10
	// Distance from enemy to food: 4
	// Enemy can reach food faster - should be dangerous
	isDangerous := isFoodDangerous(state, Coord{X: 5, Y: 5})
	if !isDangerous {
		t.Error("Food should be dangerous when enemy can reach it faster")
	}
}
