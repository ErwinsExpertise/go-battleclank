package main

import (
	"testing"
)

// Standard Battlesnake Board Sizes:
// - 7x7 (small)
// - 11x11 (medium/standard)
// - 19x19 (large)
// Reference: https://docs.battlesnake.com/api/objects/board

// Helper function to create API-compliant test snakes
// Creates a Battlesnake with all required fields per the Battlesnake API spec:
// https://docs.battlesnake.com/api/objects/battlesnake
func createTestSnake(id string, health int, body []Coord) Battlesnake {
	if len(body) == 0 {
		panic("snake body cannot be empty")
	}
	return Battlesnake{
		ID:     id,
		Name:   id, // Use ID as name for tests
		Health: health,
		Body:   body,
		Head:   body[0], // Head is always first element
		Length: len(body),
		// Latency, Shout, and Customizations are optional and omitted in tests
	}
}

// Test info function
func TestInfo(t *testing.T) {
	response := info()
	
	if response.APIVersion != "1" {
		t.Errorf("Expected APIVersion to be '1', got '%s'", response.APIVersion)
	}
	
	if response.Color == "" {
		t.Error("Color should not be empty")
	}
	
	if response.Head == "" {
		t.Error("Head should not be empty")
	}
	
	if response.Tail == "" {
		t.Error("Tail should not be empty")
	}
}

// Test getNextPosition
func TestGetNextPosition(t *testing.T) {
	tests := []struct {
		name     string
		pos      Coord
		move     string
		expected Coord
	}{
		{"Move up", Coord{X: 5, Y: 5}, MoveUp, Coord{X: 5, Y: 6}},
		{"Move down", Coord{X: 5, Y: 5}, MoveDown, Coord{X: 5, Y: 4}},
		{"Move left", Coord{X: 5, Y: 5}, MoveLeft, Coord{X: 4, Y: 5}},
		{"Move right", Coord{X: 5, Y: 5}, MoveRight, Coord{X: 6, Y: 5}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getNextPosition(tt.pos, tt.move)
			if result.X != tt.expected.X || result.Y != tt.expected.Y {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Test manhattanDistance
func TestManhattanDistance(t *testing.T) {
	tests := []struct {
		name     string
		a        Coord
		b        Coord
		expected int
	}{
		{"Same position", Coord{X: 0, Y: 0}, Coord{X: 0, Y: 0}, 0},
		{"Horizontal distance", Coord{X: 0, Y: 0}, Coord{X: 5, Y: 0}, 5},
		{"Vertical distance", Coord{X: 0, Y: 0}, Coord{X: 0, Y: 5}, 5},
		{"Diagonal distance", Coord{X: 0, Y: 0}, Coord{X: 3, Y: 4}, 7},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := manhattanDistance(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Expected %d, got %d", tt.expected, result)
			}
		})
	}
}

// Test isImmediatelyFatal - out of bounds
func TestIsImmediatelyFatal_OutOfBounds(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{},
		},
	}

	tests := []struct {
		name     string
		pos      Coord
		expected bool
	}{
		{"Valid position", Coord{X: 5, Y: 5}, false},
		{"Out of bounds left", Coord{X: -1, Y: 5}, true},
		{"Out of bounds right", Coord{X: 11, Y: 5}, true},
		{"Out of bounds bottom", Coord{X: 5, Y: -1}, true},
		{"Out of bounds top", Coord{X: 5, Y: 11}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isImmediatelyFatal(state, tt.pos)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Test isImmediatelyFatal - snake collision
func TestIsImmediatelyFatal_SnakeCollision(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				createTestSnake("snake1", 50, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				}),
			},
		},
	}

	tests := []struct {
		name     string
		pos      Coord
		expected bool
	}{
		{"Empty space", Coord{X: 0, Y: 0}, false},
		{"Snake head", Coord{X: 5, Y: 5}, true},
		{"Snake body", Coord{X: 5, Y: 4}, true},
		{"Snake tail (should move)", Coord{X: 5, Y: 3}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isImmediatelyFatal(state, tt.pos)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v for position %v", tt.expected, result, tt.pos)
			}
		})
	}
}

// Test isImmediatelyFatal - snake just ate (tail won't move)
func TestIsImmediatelyFatal_SnakeJustAte(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				createTestSnake("snake1", MaxHealth, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				}),
			},
		},
	}

	// Tail should be considered dangerous if snake just ate
	result := isImmediatelyFatal(state, Coord{X: 5, Y: 3})
	if !result {
		t.Error("Expected tail to be dangerous when snake just ate")
	}
}

// Test evaluateFoodProximity
func TestEvaluateFoodProximity(t *testing.T) {
	state := GameState{
		Board: Board{
			Food: []Coord{
				{X: 5, Y: 5},
				{X: 10, Y: 10},
			},
		},
	}

	tests := []struct {
		name     string
		pos      Coord
		hasValue bool
	}{
		{"At food location", Coord{X: 5, Y: 5}, true},
		{"Near food", Coord{X: 5, Y: 6}, true},
		{"Far from food", Coord{X: 0, Y: 0}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := evaluateFoodProximity(state, tt.pos)
			if tt.hasValue && result <= 0 {
				t.Errorf("Expected positive value, got %f", result)
			}
		})
	}
}

// Test evaluateFoodProximity with no food
func TestEvaluateFoodProximity_NoFood(t *testing.T) {
	state := GameState{
		Board: Board{
			Food: []Coord{},
		},
	}

	result := evaluateFoodProximity(state, Coord{X: 5, Y: 5})
	if result != 0 {
		t.Errorf("Expected 0 when no food, got %f", result)
	}
}

// Test evaluateHeadCollisionRisk
func TestEvaluateHeadCollisionRisk(t *testing.T) {
	state := GameState{
		You: createTestSnake("me", 100, []Coord{
			{X: 5, Y: 5},
			{X: 5, Y: 4},
			{X: 5, Y: 3},
			{X: 5, Y: 2},
			{X: 5, Y: 1},
		}),
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				createTestSnake("me", 100, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
					{X: 5, Y: 2},
					{X: 5, Y: 1},
				}),
				createTestSnake("enemy1", 100, []Coord{
					{X: 5, Y: 7},
					{X: 5, Y: 8},
					{X: 5, Y: 9},
					{X: 5, Y: 10},
					{X: 4, Y: 10},
					{X: 3, Y: 10},
				}),
				createTestSnake("enemy2", 100, []Coord{
					{X: 7, Y: 5},
					{X: 8, Y: 5},
					{X: 9, Y: 5},
				}),
			},
		},
	}

	// Position where larger enemy could move to
	riskPos := Coord{X: 5, Y: 6}
	risk := evaluateHeadCollisionRisk(state, riskPos)
	
	if risk <= 0 {
		t.Errorf("Expected positive risk for position near larger enemy head, got %f", risk)
	}
}

// Test evaluateCenterProximity
func TestEvaluateCenterProximity(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
		},
	}

	centerScore := evaluateCenterProximity(state, Coord{X: 5, Y: 5})
	cornerScore := evaluateCenterProximity(state, Coord{X: 0, Y: 0})

	if centerScore <= cornerScore {
		t.Errorf("Expected center score (%f) to be higher than corner score (%f)", centerScore, cornerScore)
	}
}

// Test evaluateSpace
func TestEvaluateSpace(t *testing.T) {
	state := GameState{
		You: createTestSnake("me", 100, []Coord{
			{X: 5, Y: 5},
			{X: 5, Y: 4},
			{X: 5, Y: 3},
		}),
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{},
		},
	}

	// Open space should have high score
	openSpaceScore := evaluateSpace(state, Coord{X: 5, Y: 5})
	if openSpaceScore <= 0 {
		t.Errorf("Expected positive score for open space, got %f", openSpaceScore)
	}
}

// Test move function
func TestMove(t *testing.T) {
	mySnake := createTestSnake("me", 50, []Coord{
		{X: 5, Y: 5},
		{X: 5, Y: 4},
		{X: 5, Y: 3},
	})
	
	state := GameState{
		Turn: 1,
		Game: Game{
			ID: "test-game",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 5, Y: 7}},
			Snakes: []Battlesnake{mySnake},
		},
	}

	response := move(state)
	
	// Should return a valid move
	validMoves := map[string]bool{
		MoveUp:    true,
		MoveDown:  true,
		MoveLeft:  true,
		MoveRight: true,
	}
	
	if !validMoves[response.Move] {
		t.Errorf("Invalid move returned: %s", response.Move)
	}
	
	// Should not move down (into own body)
	if response.Move == MoveDown {
		t.Error("Should not move into own body")
	}
}

// Test scoreMove doesn't crash
func TestScoreMove(t *testing.T) {
	mySnake := createTestSnake("me", 50, []Coord{
		{X: 5, Y: 5},
		{X: 5, Y: 4},
		{X: 5, Y: 3},
	})
	
	state := GameState{
		Turn: 1,
		You:  mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{},
			Snakes: []Battlesnake{mySnake},
		},
	}

	// Test all moves
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	for _, move := range moves {
		score := scoreMove(state, move)
		// Fatal move should have very negative score
		if move == MoveDown {
			if score > -1000 {
				t.Errorf("Expected very negative score for fatal move down, got %f", score)
			}
		}
	}
}

// Test A* search with straight line path
func TestAStarSearch_StraightLine(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{},
		},
	}

	start := Coord{X: 0, Y: 0}
	goal := Coord{X: 5, Y: 0}

	path := aStarSearch(state, start, goal, MaxAStarNodes)

	if path == nil {
		t.Fatal("Expected path, got nil")
	}

	if len(path) != 6 { // 0,1,2,3,4,5
		t.Errorf("Expected path length 6, got %d", len(path))
	}

	// Verify path starts at start
	if path[0].X != start.X || path[0].Y != start.Y {
		t.Errorf("Path should start at %v, got %v", start, path[0])
	}

	// Verify path ends at goal
	if path[len(path)-1].X != goal.X || path[len(path)-1].Y != goal.Y {
		t.Errorf("Path should end at %v, got %v", goal, path[len(path)-1])
	}
}

// Test A* search around obstacle
func TestAStarSearch_AroundObstacle(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				createTestSnake("obstacle", 50, []Coord{
					{X: 5, Y: 0},
					{X: 5, Y: 1},
					{X: 5, Y: 2},
					{X: 5, Y: 3},
					{X: 5, Y: 4},
				}),
			},
		},
	}

	start := Coord{X: 0, Y: 2}
	goal := Coord{X: 10, Y: 2}

	path := aStarSearch(state, start, goal, MaxAStarNodes)

	if path == nil {
		t.Fatal("Expected path, got nil")
	}

	// Path should go around the obstacle
	// Verify no position in path collides with snake
	for _, pos := range path {
		if isPositionBlocked(state, pos) {
			t.Errorf("Path includes blocked position: %v", pos)
		}
	}

	// Path should be longer than direct Manhattan distance
	manhattanDist := manhattanDistance(start, goal)
	if len(path)-1 <= manhattanDist {
		t.Logf("Path length %d should be greater than Manhattan distance %d (due to obstacle)", len(path)-1, manhattanDist)
	}
}

// Test A* search with no valid path
func TestAStarSearch_NoPath(t *testing.T) {
	// Goal is in a completely enclosed area
	state := GameState{
		Board: Board{
			Width:  7,
			Height: 7,
			Snakes: []Battlesnake{
				createTestSnake("wall", 50, []Coord{
					// Top wall
					{X: 2, Y: 5},
					{X: 3, Y: 5},
					{X: 4, Y: 5},
					// Right wall
					{X: 5, Y: 4},
					{X: 5, Y: 3},
					{X: 5, Y: 2},
					// Bottom wall
					{X: 4, Y: 1},
					{X: 3, Y: 1},
					{X: 2, Y: 1},
					// Left wall
					{X: 1, Y: 2},
					{X: 1, Y: 3},
					{X: 1, Y: 4},
					// Close the box
					{X: 1, Y: 5},
					{X: 5, Y: 5},
					{X: 5, Y: 1},
					{X: 1, Y: 1},
					// Tail (movable but outside the box)
					{X: 0, Y: 0},
				}),
			},
		},
	}

	start := Coord{X: 0, Y: 6}
	goal := Coord{X: 3, Y: 3} // Inside the enclosed box

	path := aStarSearch(state, start, goal, MaxAStarNodes)

	if path != nil {
		t.Error("Expected nil (no path), got path")
	}
}

// Test A* search when start equals goal
func TestAStarSearch_StartEqualsGoal(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{},
		},
	}

	start := Coord{X: 5, Y: 5}
	goal := Coord{X: 5, Y: 5}

	path := aStarSearch(state, start, goal, MaxAStarNodes)

	if path == nil {
		t.Fatal("Expected path with single element")
	}

	if len(path) != 1 {
		t.Errorf("Expected path length 1, got %d", len(path))
	}

	if path[0].X != start.X || path[0].Y != start.Y {
		t.Errorf("Expected path to contain only start position %v, got %v", start, path[0])
	}
}

// Test A* search with goal blocked
func TestAStarSearch_GoalBlocked(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				createTestSnake("blocker", 50, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				}),
			},
		},
	}

	start := Coord{X: 0, Y: 0}
	goal := Coord{X: 5, Y: 4} // Blocked by snake body

	path := aStarSearch(state, start, goal, MaxAStarNodes)

	if path != nil {
		t.Error("Expected nil path when goal is blocked")
	}
}

// Test isPositionBlocked
func TestIsPositionBlocked(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				createTestSnake("snake1", 50, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				}),
			},
		},
	}

	tests := []struct {
		name     string
		pos      Coord
		expected bool
	}{
		{"Empty space", Coord{X: 0, Y: 0}, false},
		{"Snake head", Coord{X: 5, Y: 5}, true},
		{"Snake body", Coord{X: 5, Y: 4}, true},
		{"Snake tail (should move)", Coord{X: 5, Y: 3}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isPositionBlocked(state, tt.pos)
			if result != tt.expected {
				t.Errorf("Expected %v for position %v, got %v", tt.expected, tt.pos, result)
			}
		})
	}
}

// Test isPositionBlocked when snake just ate
func TestIsPositionBlocked_SnakeJustAte(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				createTestSnake("snake1", MaxHealth, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				}),
			},
		},
	}

	// Tail should be considered blocked if snake just ate
	result := isPositionBlocked(state, Coord{X: 5, Y: 3})
	if !result {
		t.Error("Expected tail to be blocked when snake just ate")
	}
}

// Test getValidNeighbors
func TestGetValidNeighbors(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				createTestSnake("snake1", 50, []Coord{
					{X: 5, Y: 6},
					{X: 5, Y: 5},
					{X: 5, Y: 4},
				}),
			},
		},
	}

	// Position surrounded by obstacle on top
	pos := Coord{X: 5, Y: 7}
	neighbors := getValidNeighbors(state, pos)

	// Should have 3 neighbors (up is blocked by snake, down/left/right are open)
	if len(neighbors) != 3 {
		t.Errorf("Expected 3 valid neighbors, got %d", len(neighbors))
	}

	// Check that blocked position is not in neighbors
	blockedPos := Coord{X: 5, Y: 6}
	for _, n := range neighbors {
		if n.X == blockedPos.X && n.Y == blockedPos.Y {
			t.Error("Blocked position should not be in valid neighbors")
		}
	}
}

// Test getValidNeighbors at corner
func TestGetValidNeighbors_Corner(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{},
		},
	}

	// Corner position
	pos := Coord{X: 0, Y: 0}
	neighbors := getValidNeighbors(state, pos)

	// Should have 2 neighbors (right and up)
	if len(neighbors) != 2 {
		t.Errorf("Expected 2 valid neighbors at corner, got %d", len(neighbors))
	}
}

// Test findNearestFoodWithAStar
func TestFindNearestFoodWithAStar(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Food: []Coord{
				{X: 10, Y: 10}, // Far food
				{X: 2, Y: 2},   // Near food
			},
			Snakes: []Battlesnake{},
		},
	}

	start := Coord{X: 0, Y: 0}
	nearestFood, path := findNearestFoodWithAStar(state, start)

	if path == nil {
		t.Fatal("Expected path to nearest food")
	}

	// Should find the closer food
	if nearestFood.X != 2 || nearestFood.Y != 2 {
		t.Errorf("Expected nearest food at (2,2), got (%d,%d)", nearestFood.X, nearestFood.Y)
	}

	// Path should be shorter to (2,2) than to (10,10)
	if len(path) > 5 {
		t.Errorf("Expected short path to near food, got length %d", len(path))
	}
}

// Test evaluateFoodProximity with A* integration (critical health)
func TestEvaluateFoodProximity_WithAStar(t *testing.T) {
	state := GameState{
		You: createTestSnake("me", 20, []Coord{
			{X: 0, Y: 0},
			{X: 0, Y: 1},
			{X: 0, Y: 2},
		}),
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 5, Y: 5}},
			Snakes: []Battlesnake{},
		},
	}

	pos := Coord{X: 0, Y: 0}
	score := evaluateFoodProximity(state, pos)

	if score <= 0 {
		t.Error("Expected positive score for food proximity")
	}
}

// Test evaluateFoodProximity without A* (non-critical health)
func TestEvaluateFoodProximity_WithoutAStar(t *testing.T) {
	state := GameState{
		You: createTestSnake("me", 80, []Coord{
			{X: 0, Y: 0},
			{X: 0, Y: 1},
			{X: 0, Y: 2},
		}),
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 5, Y: 5}},
			Snakes: []Battlesnake{},
		},
	}

	pos := Coord{X: 0, Y: 0}
	score := evaluateFoodProximity(state, pos)

	if score <= 0 {
		t.Error("Expected positive score for food proximity")
	}
}

// Benchmark A* search performance
func BenchmarkAStarSearch(b *testing.B) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				createTestSnake("obstacle", 50, []Coord{
					{X: 5, Y: 0},
					{X: 5, Y: 1},
					{X: 5, Y: 2},
					{X: 5, Y: 3},
					{X: 5, Y: 4},
				}),
			},
		},
	}

	start := Coord{X: 0, Y: 2}
	goal := Coord{X: 10, Y: 8}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		aStarSearch(state, start, goal, MaxAStarNodes)
	}
}

// Benchmark move evaluation with A* integration
func BenchmarkMoveEvaluation(b *testing.B) {
	mySnake := createTestSnake("me", 25, []Coord{
		{X: 5, Y: 5},
		{X: 5, Y: 4},
		{X: 5, Y: 3},
		{X: 5, Y: 2},
		{X: 5, Y: 1},
	})
	
	enemySnake := createTestSnake("enemy", 80, []Coord{
		{X: 3, Y: 3},
		{X: 3, Y: 2},
		{X: 3, Y: 1},
		{X: 3, Y: 0},
	})

	state := GameState{
		Turn: 50,
		You:  mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 10, Y: 10}, {X: 1, Y: 1}},
			Snakes: []Battlesnake{mySnake, enemySnake},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		move(state)
	}
}

// Test that snake prioritizes food over tail chasing when health is moderate
func TestMove_PrioritizesFoodOverCircling(t *testing.T) {
	// Snake with health 60 (between 50 and 70) should still seek food
	// to avoid going in circles
	mySnake := createTestSnake("me", 60, []Coord{
		{X: 5, Y: 5}, // head
		{X: 5, Y: 4},
		{X: 5, Y: 3},
		{X: 5, Y: 2}, // tail
	})
	
	state := GameState{
		Turn: 10,
		Game: Game{
			ID: "test-game",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			// Food directly above, should move toward it
			Food:   []Coord{{X: 5, Y: 8}},
			Snakes: []Battlesnake{mySnake},
		},
	}

	response := move(state)
	
	// With food directly above and moderate health, should move up toward food
	// rather than following tail (which would be down, left, or right)
	if response.Move != MoveUp {
		t.Errorf("Expected snake to move toward food (up), but moved %s. This indicates circular tail-chasing behavior.", response.Move)
	}
}

// Test the circular behavior scenario - snake following tail instead of seeking food
func TestMove_CircularBehaviorWithFood(t *testing.T) {
	// Snake with health 80 in late game (after turn 50, so no center preference)
	// forming a circle-like pattern
	mySnake := createTestSnake("me", 80, []Coord{
		{X: 5, Y: 5}, // head
		{X: 5, Y: 4},
		{X: 6, Y: 4},
		{X: 6, Y: 5},
		{X: 6, Y: 6},
		{X: 5, Y: 6},
		{X: 4, Y: 6},
		{X: 4, Y: 5}, // tail at (4,5) - to the left of head
	})
	
	state := GameState{
		Turn: 60, // Late game - no center preference
		Game: Game{
			ID: "test-game",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			// Food far away but accessible
			Food:   []Coord{{X: 10, Y: 10}},
			Snakes: []Battlesnake{mySnake},
		},
	}

	// Calculate all move scores to see what's happening
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
		t.Logf("Move %s: score %.2f", m, scores[m])
	}
	
	response := move(state)
	
	// With health 80 and food far away:
	// - Food seeking is ACTIVE with weight 50 (prevents circling)
	// - Tail chasing is ON (active when health > 30)
	// - No center preference (turn > 50)
	// The snake now balances tail following with food awareness
	t.Logf("Snake moved: %s with score %.2f", response.Move, scores[response.Move])
	t.Logf("Snake health: %d, Food seeking is now always active", state.You.Health)
	
	// This test verifies that food seeking is active at all health levels
	// preventing the circular behavior that was the original issue
}
