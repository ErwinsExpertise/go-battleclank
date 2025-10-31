package sims

import "testing"

func TestBatchTester_SmallBatch(t *testing.T) {
	config := BatchTestConfig{
		NumGames:    10,
		MaxTurns:    100,
		BoardWidth:  11,
		BoardHeight: 11,
		NumEnemies:  1,
		UseGreedy:   true,
		RandomSeed:  12345, // Fixed seed for reproducibility
	}
	
	tester := NewBatchTester(config)
	results := tester.RunBatch()
	
	if results.TotalGames != 10 {
		t.Errorf("Expected 10 games, got %d", results.TotalGames)
	}
	
	if results.Wins+results.Losses != results.TotalGames {
		t.Errorf("Wins (%d) + Losses (%d) != TotalGames (%d)", 
			results.Wins, results.Losses, results.TotalGames)
	}
	
	t.Logf("Batch test completed: %d wins, %d losses", results.Wins, results.Losses)
	t.Logf("Average turns: %.1f", results.AvgTurns)
	t.Logf("Average final length: %.1f", results.AvgFinalLength)
	
	// Print death reasons
	for reason, count := range results.DeathReasons {
		t.Logf("  %s: %d", reason, count)
	}
}

func TestBatchTester_CompareStrategies(t *testing.T) {
	baseConfig := BatchTestConfig{
		NumGames:    20,
		MaxTurns:    100,
		BoardWidth:  11,
		BoardHeight: 11,
		NumEnemies:  2,
		RandomSeed:  12345,
	}
	
	// Test greedy strategy
	greedyConfig := baseConfig
	greedyConfig.UseGreedy = true
	greedyTester := NewBatchTester(greedyConfig)
	greedyResults := greedyTester.RunBatch()
	
	// Test lookahead strategy with same seed
	lookaheadConfig := baseConfig
	lookaheadConfig.UseGreedy = false
	lookaheadTester := NewBatchTester(lookaheadConfig)
	lookaheadResults := lookaheadTester.RunBatch()
	
	t.Log("\n=== Strategy Comparison ===")
	t.Logf("Greedy Strategy:")
	t.Logf("  Wins: %d/%d (%.1f%%)", greedyResults.Wins, greedyResults.TotalGames, 
		float64(greedyResults.Wins)/float64(greedyResults.TotalGames)*100)
	t.Logf("  Avg Length: %.1f", greedyResults.AvgFinalLength)
	t.Logf("  Avg Food: %.1f", greedyResults.AvgFoodCollected)
	
	t.Logf("\nLookahead Strategy:")
	t.Logf("  Wins: %d/%d (%.1f%%)", lookaheadResults.Wins, lookaheadResults.TotalGames,
		float64(lookaheadResults.Wins)/float64(lookaheadResults.TotalGames)*100)
	t.Logf("  Avg Length: %.1f", lookaheadResults.AvgFinalLength)
	t.Logf("  Avg Food: %.1f", lookaheadResults.AvgFoodCollected)
}

func TestBatchTester_DifferentConfigurations(t *testing.T) {
	configurations := []struct {
		name       string
		numEnemies int
		boardSize  int
	}{
		{"Small board, few enemies", 1, 7},
		{"Standard board, few enemies", 2, 11},
		{"Large board, many enemies", 3, 19},
	}
	
	for _, cfg := range configurations {
		t.Run(cfg.name, func(t *testing.T) {
			config := BatchTestConfig{
				NumGames:    5,
				MaxTurns:    50,
				BoardWidth:  cfg.boardSize,
				BoardHeight: cfg.boardSize,
				NumEnemies:  cfg.numEnemies,
				UseGreedy:   true,
				RandomSeed:  12345,
			}
			
			tester := NewBatchTester(config)
			results := tester.RunBatch()
			
			t.Logf("%s: %d wins, %.1f avg length", 
				cfg.name, results.Wins, results.AvgFinalLength)
		})
	}
}
