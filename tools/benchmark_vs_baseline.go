package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/ErwinsExpertise/go-battleclank/tests/sims"
)

// BenchmarkConfig holds configuration for benchmark runs
type BenchmarkConfig struct {
	NumGames    int
	BoardWidth  int
	BoardHeight int
	MaxTurns    int
	NumSnakes   int
}

// BenchmarkResults holds aggregate results from multiple games
type BenchmarkResults struct {
	Config       BenchmarkConfig
	TotalGames   int
	Wins         int
	Losses       int
	Draws        int
	WinRate      float64
	
	// Death causes
	DeathsStarve     int
	DeathsCollision  int
	DeathsHeadToHead int
	DeathsTimeout    int
	
	// Performance metrics
	AvgTurns       float64
	AvgFinalLength float64
	AvgFoodEaten   float64
	
	// Per-game details
	Games []GameResult
}

// GameResult holds results for a single game
type GameResult struct {
	GameNumber int
	Winner     string
	Turns      int
	OurLength  int
	FoodEaten  int
	DeathCause string
}

// RunBenchmark runs a series of games and collects statistics
func RunBenchmark(config BenchmarkConfig) BenchmarkResults {
	results := BenchmarkResults{
		Config:     config,
		TotalGames: config.NumGames,
		Games:      make([]GameResult, 0, config.NumGames),
	}
	
	fmt.Printf("Running benchmark: %d games on %dx%d board\n\n", 
		config.NumGames, config.BoardWidth, config.BoardHeight)
	
	// Progress tracking
	progressInterval := config.NumGames / 10
	if progressInterval == 0 {
		progressInterval = 1
	}
	
	for i := 0; i < config.NumGames; i++ {
		if (i+1)%progressInterval == 0 || i == 0 {
			fmt.Printf("Progress: %d/%d games (%.1f%%)\n", 
				i+1, config.NumGames, float64(i+1)/float64(config.NumGames)*100)
		}
		
		gameResult := runSingleGame(config, i+1)
		results.Games = append(results.Games, gameResult)
		
		// Update counters
		switch gameResult.Winner {
		case "go-snake":
			results.Wins++
		case "baseline-snake":
			results.Losses++
		default:
			results.Draws++
		}
		
		// Track death causes
		switch gameResult.DeathCause {
		case "starve":
			results.DeathsStarve++
		case "collision":
			results.DeathsCollision++
		case "head-to-head":
			results.DeathsHeadToHead++
		case "timeout":
			results.DeathsTimeout++
		}
		
		// Accumulate metrics
		results.AvgTurns += float64(gameResult.Turns)
		results.AvgFinalLength += float64(gameResult.OurLength)
		results.AvgFoodEaten += float64(gameResult.FoodEaten)
	}
	
	// Calculate averages
	if config.NumGames > 0 {
		results.WinRate = float64(results.Wins) / float64(config.NumGames) * 100
		results.AvgTurns /= float64(config.NumGames)
		results.AvgFinalLength /= float64(config.NumGames)
		results.AvgFoodEaten /= float64(config.NumGames)
	}
	
	return results
}

// runSingleGame runs a single simulated game
func runSingleGame(config BenchmarkConfig, gameNum int) GameResult {
	// Create batch test configuration
	batchConfig := sims.BatchTestConfig{
		NumGames:    1,
		MaxTurns:    config.MaxTurns,
		BoardWidth:  config.BoardWidth,
		BoardHeight: config.BoardHeight,
		NumEnemies:  config.NumSnakes - 1, // Subtract 1 for our snake
		UseGreedy:   true,
		RandomSeed:  time.Now().UnixNano() + int64(gameNum),
		CollectDetailedLog: false,
	}
	
	// Run simulation using batch tester
	tester := sims.NewBatchTester(batchConfig)
	batchResults := tester.RunBatch()
	
	// Get the first (only) game result
	if len(batchResults.Games) == 0 {
		return GameResult{
			GameNumber: gameNum,
			Winner:     "error",
			Turns:      0,
			OurLength:  0,
			FoodEaten:  0,
			DeathCause: "no-result",
		}
	}
	
	outcome := batchResults.Games[0]
	
	// Convert to GameResult
	gameResult := GameResult{
		GameNumber: gameNum,
		Turns:      outcome.Turns,
		OurLength:  outcome.FinalLength,
		FoodEaten:  outcome.FoodCollected,
		DeathCause: outcome.DeathReason,
	}
	
	// Determine winner
	if outcome.Winner {
		gameResult.Winner = "go-snake"
	} else {
		gameResult.Winner = "baseline-snake"
	}
	
	return gameResult
}

// PrintResults displays formatted benchmark results
func PrintResults(results BenchmarkResults) {
	fmt.Println("\n================================================")
	fmt.Println("  Benchmark Results Summary")
	fmt.Println("================================================")
	
	fmt.Printf("Configuration:\n")
	fmt.Printf("  Board Size: %dx%d\n", results.Config.BoardWidth, results.Config.BoardHeight)
	fmt.Printf("  Games Played: %d\n", results.TotalGames)
	fmt.Printf("  Max Turns: %d\n\n", results.Config.MaxTurns)
	
	fmt.Printf("Win/Loss Record:\n")
	fmt.Printf("  Wins:   %d (%.1f%%)\n", results.Wins, float64(results.Wins)/float64(results.TotalGames)*100)
	fmt.Printf("  Losses: %d (%.1f%%)\n", results.Losses, float64(results.Losses)/float64(results.TotalGames)*100)
	fmt.Printf("  Draws:  %d (%.1f%%)\n\n", results.Draws, float64(results.Draws)/float64(results.TotalGames)*100)
	
	fmt.Printf("Overall Win Rate: %.1f%%\n\n", results.WinRate)
	
	fmt.Printf("Cause of Death (Our Snake):\n")
	fmt.Printf("  Starvation:    %d\n", results.DeathsStarve)
	fmt.Printf("  Collision:     %d\n", results.DeathsCollision)
	fmt.Printf("  Head-to-head:  %d\n", results.DeathsHeadToHead)
	fmt.Printf("  Timeout:       %d\n\n", results.DeathsTimeout)
	
	fmt.Printf("Performance Metrics:\n")
	fmt.Printf("  Average Turns Survived:  %.1f\n", results.AvgTurns)
	fmt.Printf("  Average Final Length:    %.1f\n", results.AvgFinalLength)
	fmt.Printf("  Average Food Collected:  %.1f\n\n", results.AvgFoodEaten)
	
	// Assessment
	if results.WinRate >= 80.0 {
		fmt.Println("✓ SUCCESS: Win rate >= 80% target!")
	} else if results.WinRate >= 60.0 {
		fmt.Println("⚠ PARTIAL: Win rate >= 60% (competitive parity)")
	} else {
		fmt.Println("✗ BELOW TARGET: Win rate < 60%")
	}
	
	fmt.Println("\n================================================")
}

// SaveResults saves results to JSON file
func SaveResults(results BenchmarkResults, filename string) error {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(filename, data, 0644)
}

func main() {
	// Default configuration
	config := BenchmarkConfig{
		NumGames:    100,
		BoardWidth:  11,
		BoardHeight: 11,
		MaxTurns:    500,
		NumSnakes:   2,
	}
	
	// Parse command line args (simplified)
	if len(os.Args) > 1 {
		fmt.Sscanf(os.Args[1], "%d", &config.NumGames)
	}
	
	// Run benchmark
	startTime := time.Now()
	results := RunBenchmark(config)
	duration := time.Since(startTime)
	
	// Print results
	PrintResults(results)
	
	// Save results
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("benchmark_results_%s.json", timestamp)
	if err := SaveResults(results, filename); err != nil {
		fmt.Printf("Warning: Could not save results to %s: %v\n", filename, err)
	} else {
		fmt.Printf("Results saved to: %s\n", filename)
	}
	
	fmt.Printf("Total benchmark time: %v\n\n", duration)
	
	// Exit code based on success
	if results.WinRate >= 80.0 {
		os.Exit(0)
	} else if results.WinRate >= 60.0 {
		os.Exit(0)
	} else {
		os.Exit(1)
	}
}
