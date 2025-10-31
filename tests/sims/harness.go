package sims

import (
	"fmt"
	"github.com/ErwinsExpertise/go-battleclank/algorithms/search"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

// TestScenario represents a single test scenario
type TestScenario struct {
	Name          string
	State         *board.GameState
	ExpectedMove  string // Optional: expected move for validation
	ShouldSurvive bool   // Whether the snake should survive this move
}

// TestResult represents the outcome of running a test scenario
type TestResult struct {
	Scenario     string
	ActualMove   string
	ExpectedMove string
	Score        float64
	Passed       bool
	Message      string
}

// TestHarness runs multiple test scenarios
type TestHarness struct {
	Strategy  SearchStrategy
	Scenarios []TestScenario
}

// SearchStrategy interface for different search implementations
type SearchStrategy interface {
	FindBestMove(state *board.GameState) string
	ScoreMove(state *board.GameState, move string) float64
}

// NewTestHarness creates a new test harness
func NewTestHarness(useGreedy bool) *TestHarness {
	var strategy SearchStrategy
	if useGreedy {
		strategy = search.NewGreedySearch()
	} else {
		strategy = search.NewLookaheadSearch(2)
	}
	
	return &TestHarness{
		Strategy:  strategy,
		Scenarios: []TestScenario{},
	}
}

// AddScenario adds a test scenario
func (h *TestHarness) AddScenario(scenario TestScenario) {
	h.Scenarios = append(h.Scenarios, scenario)
}

// RunAll runs all test scenarios
func (h *TestHarness) RunAll() []TestResult {
	results := make([]TestResult, 0, len(h.Scenarios))
	
	for _, scenario := range h.Scenarios {
		result := h.RunScenario(scenario)
		results = append(results, result)
	}
	
	return results
}

// RunScenario runs a single test scenario
func (h *TestHarness) RunScenario(scenario TestScenario) TestResult {
	actualMove := h.Strategy.FindBestMove(scenario.State)
	score := h.Strategy.ScoreMove(scenario.State, actualMove)
	
	result := TestResult{
		Scenario:     scenario.Name,
		ActualMove:   actualMove,
		ExpectedMove: scenario.ExpectedMove,
		Score:        score,
		Passed:       true,
	}
	
	// Validate if expected move is provided
	if scenario.ExpectedMove != "" && actualMove != scenario.ExpectedMove {
		result.Passed = false
		result.Message = fmt.Sprintf("Expected %s, got %s", scenario.ExpectedMove, actualMove)
	}
	
	// Check survival if specified
	if scenario.ShouldSurvive {
		// Basic survival check: move should not be immediately fatal
		nextPos := board.GetNextPosition(scenario.State.You.Head, actualMove)
		if !scenario.State.Board.IsInBounds(nextPos) || scenario.State.Board.IsOccupied(nextPos, true) {
			result.Passed = false
			result.Message = "Move leads to immediate death"
		}
	}
	
	if result.Passed {
		result.Message = "Test passed"
	}
	
	return result
}

// PrintResults prints test results in a readable format
func PrintResults(results []TestResult) {
	passed := 0
	total := len(results)
	
	fmt.Println("Test Results:")
	fmt.Println("=============")
	
	for _, result := range results {
		status := "✓ PASS"
		if !result.Passed {
			status = "✗ FAIL"
		} else {
			passed++
		}
		
		fmt.Printf("%s - %s: %s (score: %.2f)\n", 
			status, result.Scenario, result.Message, result.Score)
		
		if result.ExpectedMove != "" {
			fmt.Printf("      Expected: %s, Actual: %s\n", 
				result.ExpectedMove, result.ActualMove)
		}
	}
	
	fmt.Println("=============")
	fmt.Printf("Results: %d/%d passed (%.1f%%)\n", 
		passed, total, float64(passed)/float64(total)*100)
}

// CreateBasicSurvivalScenarios creates a set of basic survival test scenarios
func CreateBasicSurvivalScenarios() []TestScenario {
	return []TestScenario{
		{
			Name: "Avoid wall collision",
			State: &board.GameState{
				Board: board.Board{
					Width:  11,
					Height: 11,
					Food:   []board.Coord{{X: 5, Y: 5}},
					Snakes: []board.Snake{
						{
							ID:     "you",
							Health: 50,
							Body:   []board.Coord{{X: 0, Y: 0}, {X: 0, Y: 1}},
							Head:   board.Coord{X: 0, Y: 0},
							Length: 2,
						},
					},
				},
				You: board.Snake{
					ID:     "you",
					Health: 50,
					Body:   []board.Coord{{X: 0, Y: 0}, {X: 0, Y: 1}},
					Head:   board.Coord{X: 0, Y: 0},
					Length: 2,
				},
			},
			ShouldSurvive: true,
		},
		{
			Name: "Seek food when hungry",
			State: &board.GameState{
				Board: board.Board{
					Width:  11,
					Height: 11,
					Food:   []board.Coord{{X: 6, Y: 5}},
					Snakes: []board.Snake{
						{
							ID:     "you",
							Health: 20,
							Body:   []board.Coord{{X: 5, Y: 5}, {X: 5, Y: 4}},
							Head:   board.Coord{X: 5, Y: 5},
							Length: 2,
						},
					},
				},
				You: board.Snake{
					ID:     "you",
					Health: 20,
					Body:   []board.Coord{{X: 5, Y: 5}, {X: 5, Y: 4}},
					Head:   board.Coord{X: 5, Y: 5},
					Length: 2,
				},
			},
			ExpectedMove:  board.MoveRight,
			ShouldSurvive: true,
		},
	}
}
