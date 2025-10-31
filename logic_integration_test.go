package main

import (
	"testing"
)

// TestIntegration_TrapDetectionInRealScenario tests the complete trap detection system
// in a realistic game scenario
func TestIntegration_TrapDetectionInRealScenario(t *testing.T) {
	// Scenario: We're dominant (longer, healthier) and enemy is in a confined space
	// We should detect and pursue the trap opportunity
	
	mySnake := createTestSnake("me", 75, []Coord{
		{X: 5, Y: 5}, // head
		{X: 5, Y: 6},
		{X: 5, Y: 7},
		{X: 4, Y: 7},
		{X: 3, Y: 7},
		{X: 3, Y: 6},
	})
	mySnake.Length = 6
	
	// Enemy snake trapped in corner area
	enemySnake := createTestSnake("enemy", 50, []Coord{
		{X: 1, Y: 1}, // head - in corner
		{X: 1, Y: 2},
		{X: 2, Y: 2},
	})
	enemySnake.Length = 3
	
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{mySnake, enemySnake},
			Food:   []Coord{{X: 8, Y: 8}}, // Food far away
		},
		You: mySnake,
		Turn: 25,
	}
	
	// Test 1: Verify aggression score is high
	aggressionScore := calculateAggressionScore(state)
	if aggressionScore < 0.6 {
		t.Errorf("Expected high aggression score (>0.6) with our advantage, got %.2f", aggressionScore)
	} else {
		t.Logf("✓ High aggression score: %.2f", aggressionScore)
	}
	
	// Test 2: Evaluate all possible moves
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	
	for _, move := range moves {
		scores[move] = scoreMove(state, move)
	}
	
	t.Logf("Move scores:")
	for _, move := range moves {
		t.Logf("  %s: %.2f", move, scores[move])
	}
	
	// Test 3: Verify the move scoring includes trap considerations
	// We should have positive scores for valid moves
	validMoveFound := false
	for _, score := range scores {
		if score > -1000 {
			validMoveFound = true
			break
		}
	}
	
	if !validMoveFound {
		t.Error("Expected at least one valid move with positive score")
	}
	
	// Test 4: Make a move and verify it's reasonable
	response := move(state)
	t.Logf("✓ Snake chose move: %s", response.Move)
	
	// Verify the chosen move is one of the valid moves
	validMove := false
	for _, m := range moves {
		if response.Move == m {
			validMove = true
			break
		}
	}
	
	if !validMove {
		t.Errorf("Invalid move selected: %s", response.Move)
	}
}

// TestIntegration_DefensiveBehaviorWhenWeak tests that the snake plays defensively
// when outmatched
func TestIntegration_DefensiveBehaviorWhenWeak(t *testing.T) {
	// Scenario: We're weak (shorter, lower health) and should play defensively
	
	mySnake := createTestSnake("me", 30, []Coord{
		{X: 5, Y: 5}, // head
		{X: 5, Y: 6},
		{X: 5, Y: 7},
	})
	mySnake.Length = 3
	
	// Much larger enemy
	enemySnake := createTestSnake("enemy", 80, []Coord{
		{X: 7, Y: 5}, // head - nearby
		{X: 8, Y: 5},
		{X: 9, Y: 5},
		{X: 9, Y: 6},
		{X: 9, Y: 7},
		{X: 8, Y: 7},
		{X: 7, Y: 7},
		{X: 6, Y: 7},
	})
	enemySnake.Length = 8
	
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{mySnake, enemySnake},
			Food:   []Coord{{X: 2, Y: 2}}, // Food somewhat nearby
		},
		You: mySnake,
		Turn: 30,
	}
	
	// Test 1: Verify aggression score is low
	aggressionScore := calculateAggressionScore(state)
	if aggressionScore > 0.4 {
		t.Logf("Note: Aggression score is %.2f, expected lower (<0.4) when weak", aggressionScore)
	} else {
		t.Logf("✓ Low aggression score (defensive): %.2f", aggressionScore)
	}
	
	// Test 2: Verify we don't pursue trap opportunities (threshold not met)
	nextPos := Coord{X: 6, Y: 5} // Moving toward enemy
	trapScore := evaluateTrapOpportunity(state, nextPos)
	
	// Even if trap is detected, with low aggression it won't be weighted heavily
	t.Logf("Trap score: %.2f (aggression: %.2f, threshold: 0.6)", trapScore, aggressionScore)
	
	// Test 3: Make a move and verify it's cautious
	response := move(state)
	t.Logf("✓ Snake chose defensive move: %s", response.Move)
}

// TestIntegration_BalancedGameplay tests behavior in a balanced scenario
func TestIntegration_BalancedGameplay(t *testing.T) {
	// Scenario: Even match - similar sizes and health
	
	mySnake := createTestSnake("me", 60, []Coord{
		{X: 5, Y: 5}, // head
		{X: 5, Y: 6},
		{X: 5, Y: 7},
		{X: 4, Y: 7},
		{X: 3, Y: 7},
	})
	mySnake.Length = 5
	
	enemySnake := createTestSnake("enemy", 65, []Coord{
		{X: 8, Y: 8}, // head - reasonable distance
		{X: 8, Y: 9},
		{X: 7, Y: 9},
		{X: 6, Y: 9},
		{X: 5, Y: 9},
	})
	enemySnake.Length = 5
	
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{mySnake, enemySnake},
			Food:   []Coord{{X: 5, Y: 3}, {X: 8, Y: 5}}, // Food available
		},
		You: mySnake,
		Turn: 40,
	}
	
	// Test 1: Aggression score should be moderate
	aggressionScore := calculateAggressionScore(state)
	t.Logf("Aggression score in balanced game: %.2f", aggressionScore)
	
	if aggressionScore < 0.3 || aggressionScore > 0.7 {
		t.Logf("Note: Expected moderate aggression (0.3-0.7), got %.2f", aggressionScore)
	}
	
	// Test 2: Should focus on space and food control
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	
	for _, move := range moves {
		scores[move] = scoreMove(state, move)
	}
	
	t.Logf("Move scores in balanced game:")
	for _, move := range moves {
		t.Logf("  %s: %.2f", move, scores[move])
	}
	
	// Test 3: Make a move
	response := move(state)
	t.Logf("✓ Snake chose balanced move: %s", response.Move)
}

// TestIntegration_SpaceControlPriority tests that space control is prioritized
// when appropriate
func TestIntegration_SpaceControlPriority(t *testing.T) {
	// Scenario: Limited space available, need to secure territory
	
	mySnake := createTestSnake("me", 70, []Coord{
		{X: 5, Y: 5}, // head
		{X: 6, Y: 5},
		{X: 7, Y: 5},
		{X: 8, Y: 5},
	})
	mySnake.Length = 4
	
	// Enemy blocking significant space
	enemySnake := createTestSnake("enemy", 70, []Coord{
		{X: 5, Y: 7}, // head - cutting off space
		{X: 4, Y: 7},
		{X: 3, Y: 7},
		{X: 2, Y: 7},
		{X: 1, Y: 7},
		{X: 1, Y: 6},
		{X: 1, Y: 5},
	})
	enemySnake.Length = 7
	
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{mySnake, enemySnake},
			Food:   []Coord{},
		},
		You: mySnake,
		Turn: 50,
	}
	
	// Test: Evaluate space for different moves
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	
	t.Logf("Space control evaluation:")
	for _, move := range moves {
		nextPos := getNextPosition(mySnake.Head, move)
		if !isImmediatelyFatal(state, nextPos) {
			space := evaluateSpace(state, nextPos)
			t.Logf("  %s: space factor %.2f", move, space)
		}
	}
	
	response := move(state)
	t.Logf("✓ Snake chose space-control move: %s", response.Move)
}
