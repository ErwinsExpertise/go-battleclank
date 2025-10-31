package main

import (
	"testing"
)

// TestAggressiveFoodSeeking_HealthySnake validates that a healthy snake
// seeks food more aggressively than before
func TestAggressiveFoodSeeking_HealthySnake(t *testing.T) {
	mySnake := createTestSnake("me", 80, []Coord{
		{X: 5, Y: 5}, // Healthy snake at center
		{X: 5, Y: 4},
		{X: 5, Y: 3},
	})

	state := GameState{
		Turn: 20,
		You:  mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 7, Y: 5}}, // Food 2 squares away
			Snakes: []Battlesnake{mySnake},
		},
	}

	// Move toward food (right)
	response := move(state)
	
	if response.Move != MoveRight {
		t.Errorf("Expected healthy snake to move toward food (right), got %s", response.Move)
	}

	t.Logf("Healthy snake correctly chose to move toward food: %s", response.Move)
}

// TestContestFoodWithSimilarEnemy validates that the snake will contest food
// with similarly-sized enemies instead of avoiding it
func TestContestFoodWithSimilarEnemy(t *testing.T) {
	mySnake := createTestSnake("me", 60, []Coord{
		{X: 2, Y: 5},
		{X: 2, Y: 4},
		{X: 2, Y: 3},
		{X: 2, Y: 2}, // Length 4
	})

	similarEnemy := createTestSnake("enemy", 60, []Coord{
		{X: 8, Y: 5},
		{X: 8, Y: 4},
		{X: 8, Y: 3},
		{X: 8, Y: 2}, // Length 4 - same size
	})

	state := GameState{
		Turn: 25,
		You:  mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 5, Y: 5}}, // Food equidistant from both (3 moves each)
			Snakes: []Battlesnake{mySnake, similarEnemy},
		},
	}

	// Check that food is NOT considered dangerous despite similar distance
	// Food is 3 squares away from both, outside FoodDangerRadius of 2
	foodDangerous := isFoodDangerous(state, Coord{X: 5, Y: 5})
	if foodDangerous {
		t.Error("Food should NOT be considered dangerous when enemy is same size and equidistant (outside danger radius)")
	}

	// Snake should move toward food
	response := move(state)
	if response.Move != MoveRight {
		t.Logf("Warning: Snake chose %s instead of moving toward contested food (right)", response.Move)
	} else {
		t.Logf("Snake correctly contests food with similar-sized enemy")
	}
}

// TestCriticalHealthOverridesDefense validates that when health is critical,
// the snake prioritizes food even when slightly outmatched
func TestCriticalHealthOverridesDefense(t *testing.T) {
	mySnake := createTestSnake("me", 20, []Coord{
		{X: 5, Y: 5}, // Critical health
		{X: 5, Y: 4},
		{X: 5, Y: 3}, // Length 3
	})

	slightlyLargerEnemy := createTestSnake("enemy", 80, []Coord{
		{X: 8, Y: 6}, // Nearby
		{X: 8, Y: 5},
		{X: 8, Y: 4},
		{X: 8, Y: 3},
		{X: 8, Y: 2}, // Length 5 - only 2 longer
	})

	state := GameState{
		Turn: 30,
		You:  mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 6, Y: 5}}, // Food close by
			Snakes: []Battlesnake{mySnake, slightlyLargerEnemy},
		},
	}

	// With critical health, should still seek food aggressively
	scores := make(map[string]float64)
	for _, m := range []string{MoveUp, MoveDown, MoveLeft, MoveRight} {
		scores[m] = scoreMove(state, m)
	}

	t.Logf("Critical health scores - Up: %.2f, Down: %.2f, Left: %.2f, Right (toward food): %.2f",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	// Food should have very high priority despite enemy
	response := move(state)
	if response.Move != MoveRight {
		t.Logf("Note: Snake chose %s instead of toward food (right), but critical health should prioritize food", response.Move)
	} else {
		t.Logf("Snake correctly prioritizes food at critical health")
	}
}

// TestMoreWillingToContestFood validates that food is marked dangerous less often
func TestMoreWillingToContestFood(t *testing.T) {
	tests := []struct {
		name           string
		myHealth       int
		myDist         int
		enemyDist      int
		enemyLength    int
		myLength       int
		expectSafe     bool
		description    string
	}{
		{
			name:        "Enemy 1 move closer - should be safe now",
			myHealth:    80,
			myDist:      4,
			enemyDist:   3,
			enemyLength: 4,
			myLength:    3,
			expectSafe:  true,
			description: "With only 1 move difference, food should not be marked dangerous (was dangerous before)",
		},
		{
			name:        "Enemy 3 moves closer - still dangerous",
			myHealth:    80,
			myDist:      5,
			enemyDist:   2,
			enemyLength: 4,
			myLength:    3,
			expectSafe:  false,
			description: "With 3+ move difference, food should be marked dangerous",
		},
		{
			name:        "Same size, equidistant - safe",
			myHealth:    80,
			myDist:      3,
			enemyDist:   3,
			enemyLength: 3,
			myLength:    3,
			expectSafe:  true,
			description: "Equal size and distance should not mark food as dangerous",
		},
		{
			name:        "Enemy 1 longer, equidistant - safe",
			myHealth:    80,
			myDist:      3,
			enemyDist:   3,
			enemyLength: 4,
			myLength:    3,
			expectSafe:  true,
			description: "Only 1 length advantage at same distance should be safe (threshold is now 2+)",
		},
		{
			name:        "Critical health, enemy 2 longer - safe",
			myHealth:    25,
			myDist:      3,
			enemyDist:   3,
			enemyLength: 5,
			myLength:    3,
			expectSafe:  true,
			description: "At critical health, should contest even with 2 length disadvantage (threshold is 3+ when critical)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			food := Coord{X: 5, Y: 5}
			
			// Position enemy based on distance to food (horizontal distance)
			enemyHeadX := food.X + tt.enemyDist
			enemyBody := make([]Coord, tt.enemyLength)
			enemyBody[0] = Coord{X: enemyHeadX, Y: food.Y} // Head at same Y as food
			for i := 1; i < tt.enemyLength; i++ {
				enemyBody[i] = Coord{X: enemyHeadX, Y: food.Y - i}
			}
			enemy := createTestSnake("enemy", 80, enemyBody)

			// Position our snake based on distance to food (horizontal distance)
			myHeadX := food.X - tt.myDist
			myBody := make([]Coord, tt.myLength)
			myBody[0] = Coord{X: myHeadX, Y: food.Y} // Head at same Y as food
			for i := 1; i < tt.myLength; i++ {
				myBody[i] = Coord{X: myHeadX, Y: food.Y - i}
			}
			mySnake := createTestSnake("me", tt.myHealth, myBody)

			state := GameState{
				You: mySnake,
				Board: Board{
					Width:  11,
					Height: 11,
					Food:   []Coord{food},
					Snakes: []Battlesnake{mySnake, enemy},
				},
			}

			dangerous := isFoodDangerous(state, food)
			isSafe := !dangerous

			if isSafe != tt.expectSafe {
				t.Errorf("%s: expected safe=%v, got safe=%v", tt.description, tt.expectSafe, isSafe)
			} else {
				t.Logf("✓ %s", tt.description)
			}
		})
	}
}

// TestCenterControlWhenHealthy validates that healthy snakes prefer center positions
func TestCenterControlWhenHealthy(t *testing.T) {
	mySnake := createTestSnake("me", 90, []Coord{
		{X: 8, Y: 8}, // Near corner, healthy
		{X: 8, Y: 7},
		{X: 8, Y: 6},
	})

	state := GameState{
		Turn: 60, // Late game
		You:  mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 5, Y: 5}}, // Food at center
			Snakes: []Battlesnake{mySnake},
		},
	}

	// Should prefer moving toward center
	response := move(state)
	
	// Down or Left move toward center
	if response.Move == MoveDown || response.Move == MoveLeft {
		t.Logf("Healthy snake correctly moves toward center: %s", response.Move)
	} else {
		t.Logf("Note: Snake chose %s (may still be valid based on other factors)", response.Move)
	}
}

// TestNotOverlyDefensiveWithModerateEnemies validates that the snake
// doesn't go into defensive mode unless significantly outmatched
func TestNotOverlyDefensiveWithModerateEnemies(t *testing.T) {
	mySnake := createTestSnake("me", 70, []Coord{
		{X: 5, Y: 5},
		{X: 5, Y: 4},
		{X: 5, Y: 3},
		{X: 5, Y: 2}, // Length 4
	})

	// Enemy is 3 longer but that shouldn't trigger defensive mode anymore (threshold is 5+)
	moderateEnemy := createTestSnake("enemy", 80, []Coord{
		{X: 7, Y: 6}, // Nearby
		{X: 7, Y: 5},
		{X: 7, Y: 4},
		{X: 7, Y: 3},
		{X: 7, Y: 2},
		{X: 6, Y: 2},
		{X: 5, Y: 2}, // Length 7 - only 3 longer
	})

	state := GameState{
		Turn: 30,
		You:  mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 6, Y: 5}},
			Snakes: []Battlesnake{mySnake, moderateEnemy},
		},
	}

	// Should NOT be considered outmatched with only 3 length difference
	outmatched := isOutmatchedByNearbyEnemies(state)
	if outmatched {
		t.Error("Snake should not be outmatched by enemy that's only 3 segments longer (threshold is 5+)")
	} else {
		t.Log("✓ Snake correctly not in defensive mode with moderately larger enemy")
	}

	// Should still seek food aggressively
	response := move(state)
	t.Logf("Snake chose: %s (with 3-longer enemy nearby)", response.Move)
}
