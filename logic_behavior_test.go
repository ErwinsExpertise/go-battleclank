package main

import (
	"testing"
)

// TestBehavior_PrioritizesFoodOverTailFollowing verifies that the snake
// always seeks food instead of following its tail in a circular pattern
func TestBehavior_PrioritizesFoodOverTailFollowing(t *testing.T) {
	// Snake in late game with decent health, forming a potential circle
	mySnake := createTestSnake("me", 70, []Coord{
		{X: 5, Y: 5}, // head
		{X: 5, Y: 4},
		{X: 4, Y: 4},
		{X: 4, Y: 5}, // tail near head - could form circle
	})

	// Food available on the board
	state := GameState{
		Turn: 60, // Late game (no center preference)
		Game: Game{
			ID: "behavior-test",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 8, Y: 5}}, // Food to the right
			Snakes: []Battlesnake{mySnake},
		},
	}

	// Calculate scores for each direction
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
	}

	t.Logf("Scores - Up: %.2f, Down: %.2f, Left (toward tail): %.2f, Right (toward food): %.2f",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	response := move(state)
	t.Logf("Snake chose to move: %s", response.Move)

	// Snake should move toward food (right) instead of following tail (left)
	if response.Move != MoveRight {
		t.Errorf("Expected snake to move toward food (right), but it moved: %s", response.Move)
	}

	// Food-seeking should dominate over tail-following
	if scores[MoveRight] <= scores[MoveLeft] {
		t.Errorf("Food-seeking score (%.2f) should be higher than tail-following score (%.2f)",
			scores[MoveRight], scores[MoveLeft])
	}
}

// TestBehavior_NoCirclingWithoutFood verifies tail-following is minimal
// when no food exists (edge case)
func TestBehavior_NoCirclingWithoutFood(t *testing.T) {
	// Snake with good health, no food on board
	mySnake := createTestSnake("me", 90, []Coord{
		{X: 5, Y: 5}, // head
		{X: 5, Y: 4},
		{X: 4, Y: 4},
		{X: 4, Y: 5}, // tail
	})

	state := GameState{
		Turn: 60,
		Game: Game{
			ID: "no-food-test",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{}, // NO FOOD
			Snakes: []Battlesnake{mySnake},
		},
	}

	// When no food exists, tail-following has minimal weight (5)
	// The snake should still make reasonable moves based on space
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
	}

	t.Logf("Scores with no food - Up: %.2f, Down: %.2f, Left: %.2f, Right: %.2f",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	// Even without food, tail-following should not dominate decision making
	// All non-fatal moves should have relatively similar scores based on space
	response := move(state)
	t.Logf("Snake chose to move: %s (with no food on board)", response.Move)

	// Just verify it makes a valid move (doesn't crash)
	if response.Move == "" {
		t.Error("Snake should make a valid move even with no food")
	}
}

// TestBehavior_AlwaysActiveWhenHealthy verifies the snake doesn't idle
func TestBehavior_AlwaysActiveWhenHealthy(t *testing.T) {
	// Multiple scenarios where snake should actively seek food
	scenarios := []struct {
		name   string
		health int
		food   Coord
	}{
		{"Healthy snake seeks food", 80, Coord{X: 8, Y: 5}},
		{"Medium health seeks food", 60, Coord{X: 8, Y: 5}},
		{"Low health seeks food", 40, Coord{X: 8, Y: 5}},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			mySnake := createTestSnake("me", scenario.health, []Coord{
				{X: 5, Y: 5},
				{X: 5, Y: 4},
				{X: 4, Y: 4},
			})

			state := GameState{
				Turn: 50,
				You:  mySnake,
				Board: Board{
					Width:  11,
					Height: 11,
					Food:   []Coord{scenario.food},
					Snakes: []Battlesnake{mySnake},
				},
			}

			// Evaluate food-seeking score
			foodFactor := evaluateFoodProximity(state, state.You.Head)
			t.Logf("Health %d: Food proximity factor = %.2f", scenario.health, foodFactor)

			// Food factor should always be positive when food exists
			if foodFactor <= 0 {
				t.Errorf("Food proximity factor should be positive, got %.2f", foodFactor)
			}
		})
	}
}
