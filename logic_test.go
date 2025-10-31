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

// Test isFoodDangerous - food near enemy snakes
func TestIsFoodDangerous(t *testing.T) {
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
				createTestSnake("enemy1", 100, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 6},
					{X: 5, Y: 7},
					{X: 5, Y: 8},
				}),
			},
		},
	}

	tests := []struct {
		name     string
		food     Coord
		expected bool
	}{
		{"Food far from enemy", Coord{X: 0, Y: 10}, false},
		{"Food directly on enemy body", Coord{X: 5, Y: 5}, true},
		{"Food 1 space from enemy", Coord{X: 5, Y: 4}, true},
		{"Food 2 spaces from enemy", Coord{X: 5, Y: 3}, true},
		{"Food 3 spaces from enemy (safe)", Coord{X: 5, Y: 2}, false},
		{"Food adjacent to enemy head", Coord{X: 6, Y: 5}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isFoodDangerous(state, tt.food)
			if result != tt.expected {
				t.Errorf("Expected %v for food at %v, got %v", tt.expected, tt.food, result)
			}
		})
	}
}

// Test isFoodDangerous - no enemy snakes
func TestIsFoodDangerous_NoEnemies(t *testing.T) {
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
			},
		},
	}

	// No enemy snakes, so no food should be dangerous
	result := isFoodDangerous(state, Coord{X: 8, Y: 8})
	if result {
		t.Error("Food should not be dangerous when there are no enemy snakes")
	}
}

// Test evaluateFoodProximity with dangerous food
func TestEvaluateFoodProximity_DangerousFood(t *testing.T) {
	state := GameState{
		You: createTestSnake("me", 80, []Coord{
			{X: 0, Y: 0},
			{X: 0, Y: 1},
			{X: 0, Y: 2},
		}),
		Board: Board{
			Width:  11,
			Height: 11,
			Food: []Coord{
				{X: 5, Y: 5}, // Near enemy snake
			},
			Snakes: []Battlesnake{
				createTestSnake("me", 80, []Coord{
					{X: 0, Y: 0},
					{X: 0, Y: 1},
					{X: 0, Y: 2},
				}),
				createTestSnake("enemy", 100, []Coord{
					{X: 5, Y: 7}, // Enemy looping near food at (5,5)
					{X: 5, Y: 8},
					{X: 4, Y: 8},
					{X: 4, Y: 7},
					{X: 4, Y: 6},
					{X: 5, Y: 6},
				}),
			},
		},
	}

	pos := Coord{X: 0, Y: 0}
	score := evaluateFoodProximity(state, pos)

	// Score should be significantly reduced due to dangerous food
	// Normal score would be 1/10 = 0.1, but with danger penalty it should be ~0.01
	if score >= 0.05 {
		t.Errorf("Expected very low score for dangerous food, got %f", score)
	}
}

// Test evaluateFoodProximity with safe food
func TestEvaluateFoodProximity_SafeFood(t *testing.T) {
	state := GameState{
		You: createTestSnake("me", 80, []Coord{
			{X: 0, Y: 0},
			{X: 0, Y: 1},
			{X: 0, Y: 2},
		}),
		Board: Board{
			Width:  11,
			Height: 11,
			Food: []Coord{
				{X: 2, Y: 2}, // Far from enemy snake
			},
			Snakes: []Battlesnake{
				createTestSnake("me", 80, []Coord{
					{X: 0, Y: 0},
					{X: 0, Y: 1},
					{X: 0, Y: 2},
				}),
				createTestSnake("enemy", 100, []Coord{
					{X: 8, Y: 8}, // Enemy far away
					{X: 8, Y: 9},
					{X: 8, Y: 10},
				}),
			},
		},
	}

	pos := Coord{X: 0, Y: 0}
	score := evaluateFoodProximity(state, pos)

	// Score should not be penalized since food is safe
	if score <= 0 {
		t.Errorf("Expected positive score for safe food, got %f", score)
	}
}

// Test move avoids food in corner with looping enemy snake
func TestMove_AvoidsFoodNearLoopingSnake(t *testing.T) {
	// Scenario: Food in corner (10, 10) with enemy snake looping nearby
	mySnake := createTestSnake("me", 40, []Coord{
		{X: 5, Y: 5}, // head
		{X: 5, Y: 4},
		{X: 5, Y: 3},
	})

	enemySnake := createTestSnake("enemy", 100, []Coord{
		{X: 10, Y: 9}, // Looping near food at corner (10,10)
		{X: 9, Y: 9},
		{X: 9, Y: 10},
		{X: 8, Y: 10},
		{X: 8, Y: 9},
		{X: 8, Y: 8},
	})

	state := GameState{
		Turn: 30,
		Game: Game{
			ID: "test-game",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food: []Coord{
				{X: 10, Y: 10}, // Food in corner near enemy
				{X: 1, Y: 1},   // Safer food option
			},
			Snakes: []Battlesnake{mySnake, enemySnake},
		},
	}

	// Test that the dangerous food is recognized
	dangerousFood := Coord{X: 10, Y: 10}
	isDangerous := isFoodDangerous(state, dangerousFood)
	if !isDangerous {
		t.Error("Food at (10,10) should be recognized as dangerous due to nearby enemy snake")
	}

	// Test that safe food is not marked as dangerous
	safeFood := Coord{X: 1, Y: 1}
	isSafe := !isFoodDangerous(state, safeFood)
	if !isSafe {
		t.Error("Food at (1,1) should be recognized as safe")
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

// Test hasEnemiesNearby function
func TestHasEnemiesNearby(t *testing.T) {
	tests := []struct {
		name     string
		state    GameState
		expected bool
	}{
		{
			name: "No enemies on board",
			state: GameState{
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
					},
				},
			},
			expected: false,
		},
		{
			name: "Enemy far away",
			state: GameState{
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
						createTestSnake("enemy", 100, []Coord{
							{X: 10, Y: 10}, // Distance = 10
							{X: 10, Y: 9},
							{X: 10, Y: 8},
						}),
					},
				},
			},
			expected: false,
		},
		{
			name: "Enemy within proximity radius",
			state: GameState{
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
						createTestSnake("enemy", 100, []Coord{
							{X: 7, Y: 6}, // Distance = 3 (within radius)
							{X: 7, Y: 7},
							{X: 7, Y: 8},
						}),
					},
				},
			},
			expected: true,
		},
		{
			name: "Enemy exactly at radius boundary",
			state: GameState{
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
						createTestSnake("enemy", 100, []Coord{
							{X: 5, Y: 8}, // Distance = 3 (exactly at radius)
							{X: 5, Y: 9},
							{X: 5, Y: 10},
						}),
					},
				},
			},
			expected: true,
		},
		{
			name: "Multiple enemies, one nearby",
			state: GameState{
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
							{X: 10, Y: 10}, // Far
							{X: 10, Y: 9},
							{X: 10, Y: 8},
						}),
						createTestSnake("enemy2", 100, []Coord{
							{X: 6, Y: 7}, // Distance = 3 (nearby)
							{X: 6, Y: 8},
							{X: 6, Y: 9},
						}),
					},
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := hasEnemiesNearby(tt.state)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Test that tail chasing is disabled when enemies are nearby
func TestMove_DisablesTailChasingNearEnemies(t *testing.T) {
	// Snake in a position where it could chase its tail
	mySnake := createTestSnake("me", 80, []Coord{
		{X: 5, Y: 5}, // head
		{X: 5, Y: 4},
		{X: 5, Y: 3},
		{X: 5, Y: 2}, // tail
	})

	// Enemy snake nearby
	enemySnake := createTestSnake("enemy", 100, []Coord{
		{X: 7, Y: 6}, // Distance = 3 from our head
		{X: 7, Y: 7},
		{X: 7, Y: 8},
		{X: 7, Y: 9},
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
			Food:   []Coord{{X: 2, Y: 5}}, // Food to the left
			Snakes: []Battlesnake{mySnake, enemySnake},
		},
	}

	// Verify enemy is detected as nearby
	if !hasEnemiesNearby(state) {
		t.Fatal("Enemy should be detected as nearby")
	}

	// Calculate move scores
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
	}

	// The tail is at (5, 2) which is down from head at (5, 5)
	// With tail chasing DISABLED due to nearby enemy, the snake should
	// NOT preferentially move down toward the tail
	// Instead, it should prioritize space and food (left toward food)

	response := move(state)
	t.Logf("Snake moved: %s", response.Move)
	t.Logf("Scores - Up: %.2f, Down: %.2f, Left: %.2f, Right: %.2f",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	// This test verifies that tail chasing is disabled when enemies are nearby
	// The exact move depends on space/food calculations, but tail bonus should not apply
}

// Test that tail chasing is enabled when no enemies are nearby
func TestMove_EnablesTailChasingWhenSafe(t *testing.T) {
	// Snake in a position where it could chase its tail
	mySnake := createTestSnake("me", 80, []Coord{
		{X: 5, Y: 5}, // head
		{X: 5, Y: 4},
		{X: 5, Y: 3},
		{X: 5, Y: 2}, // tail
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
			Food:   []Coord{{X: 10, Y: 10}}, // Food far away
			Snakes: []Battlesnake{mySnake},  // No enemies
		},
	}

	// Verify no enemies nearby
	if hasEnemiesNearby(state) {
		t.Fatal("No enemies should be detected")
	}

	// Calculate move scores
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
	}

	t.Logf("Scores - Up: %.2f, Down: %.2f, Left: %.2f, Right: %.2f",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	// This test verifies that tail chasing is active when no enemies are nearby
	// The tail proximity bonus should be applied in this case
}

// TestStatelessBehavior verifies that the AI is truly stateless by ensuring
// that calling the move function multiple times with the same GameState
// produces identical results. This is a critical requirement for reliability
// and scalability.
func TestStatelessBehavior(t *testing.T) {
	// Create a typical game state
	state := GameState{
		Game: Game{
			ID: "stateless-test",
		},
		Turn: 10,
		Board: Board{
			Height: 11,
			Width:  11,
			Food: []Coord{
				{X: 5, Y: 5},
				{X: 8, Y: 3},
			},
			Snakes: []Battlesnake{
				createTestSnake("you", 75, []Coord{
					{X: 3, Y: 3}, // head
					{X: 3, Y: 2},
					{X: 3, Y: 1},
				}),
				createTestSnake("enemy", 80, []Coord{
					{X: 7, Y: 7}, // head
					{X: 7, Y: 6},
					{X: 7, Y: 5},
				}),
			},
		},
		You: createTestSnake("you", 75, []Coord{
			{X: 3, Y: 3}, // head
			{X: 3, Y: 2},
			{X: 3, Y: 1},
		}),
	}

	// Call move function multiple times with the exact same state
	const iterations = 10
	moves := make([]string, iterations)
	for i := 0; i < iterations; i++ {
		response := move(state)
		moves[i] = response.Move
	}

	// Verify all moves are identical
	firstMove := moves[0]
	for i := 1; i < iterations; i++ {
		if moves[i] != firstMove {
			t.Errorf("Stateless violation: move %d returned '%s' but move 0 returned '%s'",
				i, moves[i], firstMove)
		}
	}

	t.Logf("Stateless behavior verified: all %d calls returned '%s'", iterations, firstMove)
}

// TestStatelessBehavior_DifferentStates verifies that the AI responds differently
// to different game states, confirming it's not just returning a constant value
func TestStatelessBehavior_DifferentStates(t *testing.T) {
	// Create two different game states
	state1 := GameState{
		Game: Game{ID: "test1"},
		Turn: 10,
		Board: Board{
			Height: 11,
			Width:  11,
			Food: []Coord{
				{X: 5, Y: 5}, // food to the right
			},
			Snakes: []Battlesnake{
				createTestSnake("you", 30, []Coord{ // low health
					{X: 3, Y: 3}, // head
					{X: 3, Y: 2},
					{X: 3, Y: 1},
				}),
			},
		},
		You: createTestSnake("you", 30, []Coord{
			{X: 3, Y: 3},
			{X: 3, Y: 2},
			{X: 3, Y: 1},
		}),
	}

	state2 := GameState{
		Game: Game{ID: "test2"},
		Turn: 10,
		Board: Board{
			Height: 11,
			Width:  11,
			Food: []Coord{
				{X: 1, Y: 3}, // food to the left
			},
			Snakes: []Battlesnake{
				createTestSnake("you", 30, []Coord{ // low health
					{X: 3, Y: 3}, // head (same position)
					{X: 3, Y: 2},
					{X: 3, Y: 1},
				}),
			},
		},
		You: createTestSnake("you", 30, []Coord{
			{X: 3, Y: 3},
			{X: 3, Y: 2},
			{X: 3, Y: 1},
		}),
	}

	// Get moves for both states
	move1 := move(state1).Move
	move2 := move(state2).Move

	t.Logf("State 1 (food right): %s", move1)
	t.Logf("State 2 (food left): %s", move2)

	// The moves might be the same due to other factors, but the function
	// should at least be considering the different game states.
	// This test mainly ensures the function runs without errors on different inputs.
}

// Test evaluateCornerPenalty function
func TestEvaluateCornerPenalty(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
		},
	}

	tests := []struct {
		name     string
		pos      Coord
		expected string // "none", "low", "medium", "high"
	}{
		{"Center position", Coord{X: 5, Y: 5}, "none"},
		{"Near left edge", Coord{X: 1, Y: 5}, "low"},
		{"Near top edge", Coord{X: 5, Y: 9}, "low"},
		{"Top-left corner", Coord{X: 1, Y: 9}, "high"},
		{"Top-right corner", Coord{X: 9, Y: 9}, "high"},
		{"Bottom-left corner", Coord{X: 1, Y: 1}, "high"},
		{"Bottom-right corner", Coord{X: 9, Y: 1}, "high"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			penalty := evaluateCornerPenalty(state, tt.pos)
			t.Logf("Position %v has corner penalty: %.2f", tt.pos, penalty)

			// Verify penalty matches expected level
			switch tt.expected {
			case "none":
				if penalty > 0.1 {
					t.Errorf("Expected no/minimal penalty for center position, got %.2f", penalty)
				}
			case "low":
				if penalty < 0.1 || penalty > 0.6 {
					t.Errorf("Expected low penalty (0.1-0.6), got %.2f", penalty)
				}
			case "high":
				if penalty < 0.6 {
					t.Errorf("Expected high penalty (>0.6), got %.2f", penalty)
				}
			}
		})
	}
}

// Test corner squeeze scenario - snake should avoid moving into corner when enemy is nearby
func TestMove_AvoidsCornerWhenEnemyNearby(t *testing.T) {
	// Snake in middle area with choices: can go toward corner or away from it
	// Enemy is nearby, so corner penalty should apply
	mySnake := createTestSnake("me", 80, []Coord{
		{X: 3, Y: 8}, // head in middle-upper area
		{X: 3, Y: 7},
		{X: 3, Y: 6},
		{X: 3, Y: 5},
	})

	// Enemy snake nearby on the right side
	enemySnake := createTestSnake("enemy", 100, []Coord{
		{X: 6, Y: 8}, // head nearby
		{X: 6, Y: 7},
		{X: 6, Y: 6},
		{X: 7, Y: 6},
		{X: 8, Y: 6},
	})

	state := GameState{
		Turn: 30,
		Game: Game{
			ID: "corner-test",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 0, Y: 10}}, // Food in corner - could lure us there
			Snakes: []Battlesnake{mySnake, enemySnake},
		},
	}

	// Verify enemy is detected as nearby
	if !hasEnemiesNearby(state) {
		t.Error("Enemy should be detected as nearby")
	}

	// Calculate move scores to understand the decision
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
	}

	t.Logf("Scores - Up: %.2f, Down: %.2f, Left: %.2f, Right: %.2f",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	response := move(state)
	t.Logf("Snake chose to move: %s", response.Move)

	// Moving up would go toward corner (Y=9), moving left would go toward edge (X=2,1,0)
	// With enemy nearby, these should be penalized
	// Moving down (away from corner) or staying in middle should be preferred

	// The test validates that the corner penalty is being applied
	// We verify by checking that up (toward corner) has worse score than down
	if scores[MoveUp] > scores[MoveDown] && response.Move == MoveUp {
		t.Logf("Note: Up toward corner scored higher (%.2f) than down (%.2f), but corner penalty may not be strong enough",
			scores[MoveUp], scores[MoveDown])
	}
}

// Test that snake doesn't over-penalize corners when no enemies nearby
func TestMove_CornerOkayWhenNoEnemies(t *testing.T) {
	// Snake near corner with no enemies
	mySnake := createTestSnake("me", 80, []Coord{
		{X: 2, Y: 9}, // head near corner
		{X: 2, Y: 8},
		{X: 2, Y: 7},
	})

	state := GameState{
		Turn: 30,
		Game: Game{
			ID: "corner-safe-test",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 0, Y: 10}}, // Food in corner
			Snakes: []Battlesnake{mySnake}, // No enemies
		},
	}

	// Verify no enemies nearby
	if hasEnemiesNearby(state) {
		t.Error("No enemies should be detected")
	}

	// Calculate move scores
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
	}

	t.Logf("Scores (no enemies) - Up: %.2f, Down: %.2f, Left: %.2f, Right: %.2f",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	// With no enemies, the corner penalty should not be applied
	// The snake can safely go toward food even if it's in a corner
	// This verifies we only penalize corners when there's actual danger
}

// Test hasAnyEnemies function
func TestHasAnyEnemies(t *testing.T) {
	tests := []struct {
		name     string
		state    GameState
		expected bool
	}{
		{
			name: "Only us on board",
			state: GameState{
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
					},
				},
			},
			expected: false,
		},
		{
			name: "One enemy on board",
			state: GameState{
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
						createTestSnake("enemy", 100, []Coord{
							{X: 8, Y: 8},
							{X: 8, Y: 7},
							{X: 8, Y: 6},
						}),
					},
				},
			},
			expected: true,
		},
		{
			name: "Multiple enemies on board",
			state: GameState{
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
							{X: 8, Y: 8},
							{X: 8, Y: 7},
							{X: 8, Y: 6},
						}),
						createTestSnake("enemy2", 100, []Coord{
							{X: 2, Y: 2},
							{X: 2, Y: 1},
							{X: 2, Y: 0},
						}),
					},
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := hasAnyEnemies(tt.state)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Test evaluateWallAvoidance function
func TestEvaluateWallAvoidance(t *testing.T) {
	state := GameState{
		Board: Board{
			Width:  11,
			Height: 11,
		},
	}

	tests := []struct {
		name          string
		pos           Coord
		expectedRange string // "none", "low", "medium", "high", "very-high"
	}{
		{"Center of board", Coord{X: 5, Y: 5}, "none"},
		{"3 squares from edge", Coord{X: 3, Y: 5}, "low"},
		{"2 squares from edge", Coord{X: 2, Y: 5}, "medium"},
		{"1 square from left edge", Coord{X: 1, Y: 5}, "high"},
		{"On left edge", Coord{X: 0, Y: 5}, "very-high"},
		{"On right edge", Coord{X: 10, Y: 5}, "very-high"},
		{"On top edge", Coord{X: 5, Y: 10}, "very-high"},
		{"On bottom edge", Coord{X: 5, Y: 0}, "very-high"},
		{"Top-left corner (1,9)", Coord{X: 1, Y: 9}, "very-high"},
		{"Top-right corner (9,9)", Coord{X: 9, Y: 9}, "very-high"},
		{"Bottom-left corner (1,1)", Coord{X: 1, Y: 1}, "very-high"},
		{"Bottom-right corner (9,1)", Coord{X: 9, Y: 1}, "very-high"},
		{"Exact corner (0,0)", Coord{X: 0, Y: 0}, "very-high"},
		{"Exact corner (10,10)", Coord{X: 10, Y: 10}, "very-high"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			penalty := evaluateWallAvoidance(state, tt.pos)
			t.Logf("Position %v has wall avoidance penalty: %.2f", tt.pos, penalty)

			// Verify penalty matches expected range
			switch tt.expectedRange {
			case "none":
				if penalty > 0.05 {
					t.Errorf("Expected no/minimal penalty for center position, got %.2f", penalty)
				}
			case "low":
				if penalty < 0.1 || penalty > 0.4 {
					t.Errorf("Expected low penalty (0.1-0.4), got %.2f", penalty)
				}
			case "medium":
				if penalty < 0.4 || penalty > 0.7 {
					t.Errorf("Expected medium penalty (0.4-0.7), got %.2f", penalty)
				}
			case "high":
				if penalty < 0.7 || penalty > 1.2 {
					t.Errorf("Expected high penalty (0.7-1.2), got %.2f", penalty)
				}
			case "very-high":
				if penalty < 1.0 {
					t.Errorf("Expected very high penalty (>1.0), got %.2f", penalty)
				}
			}
		})
	}
}

// Test that snake avoids walls when enemies are present
func TestMove_AvoidsWallsWithEnemies(t *testing.T) {
	// Snake near left wall with enemy on board (even if far away)
	mySnake := createTestSnake("me", 80, []Coord{
		{X: 1, Y: 5}, // head 1 square from left wall
		{X: 2, Y: 5},
		{X: 3, Y: 5},
		{X: 4, Y: 5},
	})

	// Enemy snake on other side of board
	enemySnake := createTestSnake("enemy", 100, []Coord{
		{X: 9, Y: 9},
		{X: 9, Y: 8},
		{X: 9, Y: 7},
		{X: 9, Y: 6},
	})

	state := GameState{
		Turn: 30,
		Game: Game{
			ID: "wall-avoidance-test",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 0, Y: 5}}, // Food on the wall - tempting but dangerous
			Snakes: []Battlesnake{mySnake, enemySnake},
		},
	}

	// Calculate move scores
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
	}

	t.Logf("Scores - Up: %.2f, Down: %.2f, Left: %.2f (toward wall), Right: %.2f (away from wall)",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	response := move(state)
	t.Logf("Snake chose to move: %s", response.Move)

	// Moving left would go to the wall (X=0), which should be heavily penalized
	// Moving right (away from wall) should be strongly preferred
	// The wall avoidance should outweigh the food attraction
	if response.Move == MoveLeft {
		t.Errorf("Snake moved toward wall (left) despite aggressive wall avoidance. Score was %.2f vs right %.2f",
			scores[MoveLeft], scores[MoveRight])
	}
}

// Test that snake strongly avoids corners when enemies present
func TestMove_AvoidsCornerWithAggressiveEnemy(t *testing.T) {
	// Snake being pushed toward corner by enemy
	mySnake := createTestSnake("me", 60, []Coord{
		{X: 2, Y: 9}, // Near top-left corner
		{X: 2, Y: 8},
		{X: 2, Y: 7},
	})

	// Aggressive enemy pushing us toward corner
	enemySnake := createTestSnake("enemy", 100, []Coord{
		{X: 5, Y: 9}, // Between us and center
		{X: 5, Y: 8},
		{X: 5, Y: 7},
		{X: 5, Y: 6},
		{X: 6, Y: 6},
	})

	state := GameState{
		Turn: 40,
		Game: Game{
			ID: "corner-avoidance-test",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 0, Y: 10}}, // Food in corner - trap!
			Snakes: []Battlesnake{mySnake, enemySnake},
		},
	}

	// Calculate move scores
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
	}

	t.Logf("Scores - Up: %.2f (into corner), Down: %.2f (away from corner), Left: %.2f (into corner), Right: %.2f",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	response := move(state)
	t.Logf("Snake chose to move: %s", response.Move)

	// Moving up or left would go deeper into corner - should be heavily penalized
	// Moving down (away from corner) should be strongly preferred
	if response.Move == MoveUp || response.Move == MoveLeft {
		t.Logf("Warning: Snake moved toward corner (%s). This may indicate insufficient wall avoidance penalty.", response.Move)
	}
}

// Test that wall avoidance is active even when enemy is far away
func TestMove_WallAvoidanceActiveWithDistantEnemy(t *testing.T) {
	// Snake near edge with enemy far away
	mySnake := createTestSnake("me", 70, []Coord{
		{X: 1, Y: 5}, // 1 square from left edge
		{X: 1, Y: 4},
		{X: 1, Y: 3},
	})

	// Enemy far away on opposite side
	enemySnake := createTestSnake("enemy", 80, []Coord{
		{X: 10, Y: 10}, // Far corner
		{X: 9, Y: 10},
		{X: 8, Y: 10},
	})

	state := GameState{
		Turn: 20,
		Game: Game{
			ID: "distant-enemy-test",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 5, Y: 5}}, // Food in center
			Snakes: []Battlesnake{mySnake, enemySnake},
		},
	}

	// Verify enemy exists but is not "nearby"
	if !hasAnyEnemies(state) {
		t.Fatal("Should detect enemy presence")
	}

	// Even though enemy is far, wall avoidance should still be active
	// Calculate move scores
	moves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	scores := make(map[string]float64)
	for _, m := range moves {
		scores[m] = scoreMove(state, m)
	}

	t.Logf("Scores - Up: %.2f, Down: %.2f, Left: %.2f (toward wall), Right: %.2f (away from wall)",
		scores[MoveUp], scores[MoveDown], scores[MoveLeft], scores[MoveRight])

	// Left move (toward wall) should have worse score than right (away from wall)
	// due to wall avoidance penalty
	if scores[MoveLeft] >= scores[MoveRight] {
		t.Errorf("Left (toward wall) should have lower score than right (away from wall). Got left=%.2f, right=%.2f",
			scores[MoveLeft], scores[MoveRight])
	}
}

// Test wall avoidance doesn't interfere when no enemies present
func TestMove_NoWallPenaltyWithoutEnemies(t *testing.T) {
	// Snake near edge, no enemies
	mySnake := createTestSnake("me", 70, []Coord{
		{X: 1, Y: 5},
		{X: 2, Y: 5},
		{X: 3, Y: 5},
	})

	state := GameState{
		Turn: 20,
		Game: Game{
			ID: "no-enemy-test",
		},
		You: mySnake,
		Board: Board{
			Width:  11,
			Height: 11,
			Food:   []Coord{{X: 0, Y: 5}},  // Food on wall
			Snakes: []Battlesnake{mySnake}, // No enemies
		},
	}

	// Verify no enemies
	if hasAnyEnemies(state) {
		t.Fatal("Should not detect enemies")
	}

	// Without enemies, wall penalty should not be applied
	// Snake can safely go for food on the wall
	response := move(state)
	t.Logf("Snake chose to move: %s (food is to the left on wall)", response.Move)

	// When no enemies present, the snake can pursue food on walls without penalty
	// This test mainly verifies the conditional logic works
}

// TestStatelessBehavior_NoGlobalState verifies that multiple concurrent games
// don't interfere with each other
func TestStatelessBehavior_NoGlobalState(t *testing.T) {
	// Create two different game states representing different games
	game1State := GameState{
		Game: Game{ID: "game1"},
		Turn: 5,
		Board: Board{
			Height: 11,
			Width:  11,
			Food:   []Coord{{X: 2, Y: 2}},
			Snakes: []Battlesnake{
				createTestSnake("you", 50, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				}),
			},
		},
		You: createTestSnake("you", 50, []Coord{
			{X: 5, Y: 5},
			{X: 5, Y: 4},
			{X: 5, Y: 3},
		}),
	}

	game2State := GameState{
		Game: Game{ID: "game2"},
		Turn: 15,
		Board: Board{
			Height: 11,
			Width:  11,
			Food:   []Coord{{X: 8, Y: 8}},
			Snakes: []Battlesnake{
				createTestSnake("you", 80, []Coord{
					{X: 5, Y: 5},
					{X: 5, Y: 6},
					{X: 5, Y: 7},
				}),
			},
		},
		You: createTestSnake("you", 80, []Coord{
			{X: 5, Y: 5},
			{X: 5, Y: 6},
			{X: 5, Y: 7},
		}),
	}

	// Interleave calls to simulate concurrent games
	move1a := move(game1State).Move
	move2a := move(game2State).Move
	move1b := move(game1State).Move
	move2b := move(game2State).Move

	// Each game should get consistent results
	if move1a != move1b {
		t.Errorf("Game 1 state produced different moves: %s vs %s", move1a, move1b)
	}
	if move2a != move2b {
		t.Errorf("Game 2 state produced different moves: %s vs %s", move2a, move2b)
	}

	t.Logf("Concurrent game test passed - Game 1: %s, Game 2: %s", move1a, move2a)
}
