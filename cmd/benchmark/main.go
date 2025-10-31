package main

import (
	"flag"
	"fmt"
	"github.com/ErwinsExpertise/go-battleclank/telemetry"
	"github.com/ErwinsExpertise/go-battleclank/tests/sims"
)

// Benchmark CLI tool for running batch tests and analysis

func main() {
	// Define command-line flags
	numGames := flag.Int("games", 100, "Number of games to simulate")
	maxTurns := flag.Int("turns", 200, "Maximum turns per game")
	boardSize := flag.Int("board", 11, "Board size (width and height)")
	numEnemies := flag.Int("enemies", 2, "Number of enemy snakes")
	strategy := flag.String("strategy", "greedy", "Strategy to test: greedy or lookahead")
	seed := flag.Int64("seed", 0, "Random seed (0 for random)")
	detailed := flag.Bool("detailed", false, "Collect detailed move logs")
	compare := flag.Bool("compare", false, "Compare all strategies")
	
	flag.Parse()
	
	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Go-BattleClank Benchmark Tool                         ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	
	if *compare {
		runComparison(*numGames, *maxTurns, *boardSize, *numEnemies, *seed)
	} else {
		runSingleStrategy(*strategy, *numGames, *maxTurns, *boardSize, *numEnemies, *seed, *detailed)
	}
}

func runSingleStrategy(strategyName string, numGames, maxTurns, boardSize, numEnemies int, seed int64, detailed bool) {
	fmt.Printf("\nRunning %d games with %s strategy...\n", numGames, strategyName)
	fmt.Printf("Configuration: %dx%d board, %d enemies, max %d turns\n", 
		boardSize, boardSize, numEnemies, maxTurns)
	
	useGreedy := (strategyName == "greedy")
	
	config := sims.BatchTestConfig{
		NumGames:           numGames,
		MaxTurns:           maxTurns,
		BoardWidth:         boardSize,
		BoardHeight:        boardSize,
		NumEnemies:         numEnemies,
		UseGreedy:          useGreedy,
		RandomSeed:         seed,
		CollectDetailedLog: detailed,
	}
	
	tester := sims.NewBatchTester(config)
	results := tester.RunBatch()
	
	// Print results
	sims.PrintBatchResults(results)
	
	// Perform failure analysis
	fmt.Println("\nPerforming failure analysis...")
	analyzer := telemetry.NewFailureAnalyzer()
	
	for _, game := range results.Games {
		if !game.Winner {
			gameResult := telemetry.GameResult{
				GameID:      fmt.Sprintf("game-%d", game.GameID),
				Winner:      game.Winner,
				FinalTurn:   game.Turns,
				FinalLength: game.FinalLength,
				DeathReason: game.DeathReason,
				TotalMoves:  game.Turns,
			}
			analyzer.AddFailure(gameResult)
		}
	}
	
	analysis := analyzer.Analyze()
	telemetry.PrintAnalysis(analysis)
}

func runComparison(numGames, maxTurns, boardSize, numEnemies int, seed int64) {
	fmt.Printf("\nComparing strategies across %d games each...\n", numGames)
	fmt.Printf("Configuration: %dx%d board, %d enemies, max %d turns\n", 
		boardSize, boardSize, numEnemies, maxTurns)
	
	strategies := []struct {
		name      string
		useGreedy bool
	}{
		{"Greedy", true},
		{"Lookahead", false},
	}
	
	allResults := make(map[string]sims.BatchResults)
	
	for _, strat := range strategies {
		fmt.Printf("\n--- Testing %s Strategy ---\n", strat.name)
		
		config := sims.BatchTestConfig{
			NumGames:    numGames,
			MaxTurns:    maxTurns,
			BoardWidth:  boardSize,
			BoardHeight: boardSize,
			NumEnemies:  numEnemies,
			UseGreedy:   strat.useGreedy,
			RandomSeed:  seed,
		}
		
		tester := sims.NewBatchTester(config)
		results := tester.RunBatch()
		allResults[strat.name] = results
		
		fmt.Printf("  Wins: %d/%d (%.1f%%)\n", results.Wins, results.TotalGames,
			float64(results.Wins)/float64(results.TotalGames)*100)
		fmt.Printf("  Avg Length: %.1f\n", results.AvgFinalLength)
		fmt.Printf("  Avg Turns: %.1f\n", results.AvgTurns)
	}
	
	// Print comparison summary
	fmt.Println("\n╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║           STRATEGY COMPARISON SUMMARY                      ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	
	fmt.Printf("\n%-15s | %8s | %10s | %10s | %12s\n", 
		"Strategy", "Win Rate", "Avg Length", "Avg Turns", "Avg Food")
	fmt.Println("─────────────────┼──────────┼────────────┼────────────┼──────────────")
	
	for _, strat := range strategies {
		results := allResults[strat.name]
		winRate := float64(results.Wins) / float64(results.TotalGames) * 100
		fmt.Printf("%-15s | %6.1f%% | %10.1f | %10.1f | %12.1f\n",
			strat.name, winRate, results.AvgFinalLength, 
			results.AvgTurns, results.AvgFoodCollected)
	}
	
	// Determine best strategy
	var bestStrategy string
	var bestWinRate float64
	
	for name, results := range allResults {
		winRate := float64(results.Wins) / float64(results.TotalGames)
		if winRate > bestWinRate {
			bestWinRate = winRate
			bestStrategy = name
		}
	}
	
	fmt.Printf("\n🏆 Best Strategy: %s (%.1f%% win rate)\n", bestStrategy, bestWinRate*100)
}
