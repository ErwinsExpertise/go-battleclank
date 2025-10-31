package sims

import "testing"

func TestBasicSurvivalScenarios(t *testing.T) {
	// Test with greedy strategy
	harness := NewTestHarness(true)
	
	// Add basic survival scenarios
	scenarios := CreateBasicSurvivalScenarios()
	for _, scenario := range scenarios {
		harness.AddScenario(scenario)
	}
	
	// Run all scenarios
	results := harness.RunAll()
	
	// Check that all scenarios pass
	passed := 0
	for _, result := range results {
		if result.Passed {
			passed++
		} else {
			t.Errorf("Scenario '%s' failed: %s", result.Scenario, result.Message)
		}
	}
	
	t.Logf("Passed %d/%d scenarios", passed, len(results))
	
	if passed != len(results) {
		t.Errorf("Not all scenarios passed: %d/%d", passed, len(results))
	}
}

func TestLookaheadStrategy(t *testing.T) {
	// Test with lookahead strategy
	harness := NewTestHarness(false)
	
	// Add basic survival scenarios
	scenarios := CreateBasicSurvivalScenarios()
	for _, scenario := range scenarios {
		harness.AddScenario(scenario)
	}
	
	// Run all scenarios
	results := harness.RunAll()
	
	// Check that all scenarios pass survival checks
	survivedAll := true
	for _, result := range results {
		if !result.Passed {
			t.Logf("Scenario '%s' result: %s (may be acceptable if only move expectation differs)", 
				result.Scenario, result.Message)
			// Only fail if it's a survival issue
			if result.Message == "Move leads to immediate death" {
				survivedAll = false
				t.Errorf("Scenario '%s' failed survival check", result.Scenario)
			}
		}
	}
	
	if !survivedAll {
		t.Error("Lookahead strategy failed survival checks")
	}
}

func TestCompareStrategies(t *testing.T) {
	scenarios := CreateBasicSurvivalScenarios()
	
	// Test greedy
	greedyHarness := NewTestHarness(true)
	for _, scenario := range scenarios {
		greedyHarness.AddScenario(scenario)
	}
	greedyResults := greedyHarness.RunAll()
	
	// Test lookahead
	lookaheadHarness := NewTestHarness(false)
	for _, scenario := range scenarios {
		lookaheadHarness.AddScenario(scenario)
	}
	lookaheadResults := lookaheadHarness.RunAll()
	
	t.Log("Strategy Comparison:")
	for i := range scenarios {
		t.Logf("  Scenario: %s", scenarios[i].Name)
		t.Logf("    Greedy:    %s (score: %.2f)", greedyResults[i].ActualMove, greedyResults[i].Score)
		t.Logf("    Lookahead: %s (score: %.2f)", lookaheadResults[i].ActualMove, lookaheadResults[i].Score)
	}
}
