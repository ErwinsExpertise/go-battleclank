package sims

import (
	"fmt"
	"github.com/ErwinsExpertise/go-battleclank/algorithms/search"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/ErwinsExpertise/go-battleclank/engine/simulation"
	"math/rand"
	"time"
)

// BatchTestConfig configures batch testing parameters
type BatchTestConfig struct {
	NumGames           int
	MaxTurns           int
	BoardWidth         int
	BoardHeight        int
	NumEnemies         int
	UseGreedy          bool
	RandomSeed         int64
	CollectDetailedLog bool
}

// GameOutcome represents the result of a single game
type GameOutcome struct {
	GameID       int
	Winner       bool
	Turns        int
	FinalLength  int
	FinalHealth  int
	DeathReason  string // "collision", "starvation", "head-to-head", "trapped", "survived"
	FoodCollected int
	MovesLog     []string // Optional detailed move log
}

// BatchResults aggregates results from multiple games
type BatchResults struct {
	TotalGames      int
	Wins            int
	Losses          int
	AvgTurns        float64
	AvgFinalLength  float64
	DeathReasons    map[string]int
	AvgFoodCollected float64
	Games           []GameOutcome
}

// BatchTester runs multiple game simulations
type BatchTester struct {
	Config   BatchTestConfig
	Strategy SearchStrategy
	rng      *rand.Rand
}

// NewBatchTester creates a new batch tester
func NewBatchTester(config BatchTestConfig) *BatchTester {
	var strategy SearchStrategy
	if config.UseGreedy {
		strategy = search.NewGreedySearch()
	} else {
		strategy = search.NewLookaheadSearch(2)
	}
	
	seed := config.RandomSeed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	
	return &BatchTester{
		Config:   config,
		Strategy: strategy,
		rng:      rand.New(rand.NewSource(seed)),
	}
}

// RunBatch runs multiple game simulations
func (bt *BatchTester) RunBatch() BatchResults {
	results := BatchResults{
		TotalGames:   bt.Config.NumGames,
		DeathReasons: make(map[string]int),
		Games:        make([]GameOutcome, 0, bt.Config.NumGames),
	}
	
	for i := 0; i < bt.Config.NumGames; i++ {
		outcome := bt.runSingleGame(i + 1)
		results.Games = append(results.Games, outcome)
		
		if outcome.Winner {
			results.Wins++
		} else {
			results.Losses++
		}
		
		results.DeathReasons[outcome.DeathReason]++
	}
	
	// Calculate averages
	if results.TotalGames > 0 {
		totalTurns := 0
		totalLength := 0
		totalFood := 0
		
		for _, game := range results.Games {
			totalTurns += game.Turns
			totalLength += game.FinalLength
			totalFood += game.FoodCollected
		}
		
		results.AvgTurns = float64(totalTurns) / float64(results.TotalGames)
		results.AvgFinalLength = float64(totalLength) / float64(results.TotalGames)
		results.AvgFoodCollected = float64(totalFood) / float64(results.TotalGames)
	}
	
	return results
}

// runSingleGame simulates one complete game
func (bt *BatchTester) runSingleGame(gameID int) GameOutcome {
	// Initialize game state
	state := bt.createInitialState()
	
	outcome := GameOutcome{
		GameID:      gameID,
		MovesLog:    make([]string, 0),
		FoodCollected: 0,
	}
	
	// Run game loop
	for turn := 0; turn < bt.Config.MaxTurns; turn++ {
		// Check if our snake is still alive
		if !isSnakeAlive(state, "you") {
			outcome.Winner = false
			outcome.Turns = turn
			outcome.DeathReason = categorizeDeathReason(state, turn)
			return outcome
		}
		
		// Get our snake's current state
		ourSnake := state.Board.GetSnakeByID("you")
		if ourSnake == nil {
			outcome.Winner = false
			outcome.Turns = turn
			outcome.DeathReason = "eliminated"
			return outcome
		}
		
		outcome.FinalLength = ourSnake.Length
		outcome.FinalHealth = ourSnake.Health
		
		// Find best move
		move := bt.Strategy.FindBestMove(state)
		
		if bt.Config.CollectDetailedLog {
			outcome.MovesLog = append(outcome.MovesLog, move)
		}
		
		// Simulate move
		oldLength := ourSnake.Length
		state = simulation.SimulateMove(state, "you", move)
		newSnake := state.Board.GetSnakeByID("you")
		
		// Track food collection
		if newSnake != nil && newSnake.Length > oldLength {
			outcome.FoodCollected++
		}
		
		// Simulate enemy moves (random for now)
		state = bt.simulateEnemyMoves(state)
		
		// Remove dead snakes
		state = bt.removeDeadSnakes(state)
		
		// Check if all enemies are dead (early win)
		enemiesAlive := 0
		for _, snake := range state.Board.Snakes {
			if snake.ID != "you" && snake.Health > 0 {
				enemiesAlive++
			}
		}
		if enemiesAlive == 0 {
			// We won! All enemies eliminated
			outcome.Winner = true
			outcome.Turns = turn + 1
			outcome.DeathReason = "eliminated-all-enemies"
			return outcome
		}
		
		// Spawn food randomly
		if turn%5 == 0 && len(state.Board.Food) < 3 {
			state = bt.spawnFood(state)
		}
	}
	
	// Reached max turns - determine winner by comparing with remaining snakes
	outcome.Turns = bt.Config.MaxTurns
	outcome.DeathReason = "survived"
	
	// Check if we're the last snake alive
	aliveSnakes := 0
	ourLength := 0
	longestEnemyLength := 0
	
	for _, snake := range state.Board.Snakes {
		if snake.Health > 0 {
			aliveSnakes++
			if snake.ID == "you" {
				ourLength = snake.Length
			} else if snake.Length > longestEnemyLength {
				longestEnemyLength = snake.Length
			}
		}
	}
	
	// Win conditions:
	// 1. We're the only snake alive
	// 2. We survived and are longer than all enemies
	if aliveSnakes == 1 || ourLength > longestEnemyLength {
		outcome.Winner = true
	} else {
		outcome.Winner = false
		outcome.DeathReason = "outlasted"  // Lost on length/survival comparison
	}
	
	return outcome
}

// createInitialState creates a random initial game state
func (bt *BatchTester) createInitialState() *board.GameState {
	snakes := make([]board.Snake, bt.Config.NumEnemies+1)
	
	// Create our snake
	ourPos := bt.randomPosition()
	snakes[0] = board.Snake{
		ID:     "you",
		Name:   "go-battleclank",
		Health: 100,
		Body:   []board.Coord{ourPos, {X: ourPos.X, Y: ourPos.Y - 1}, {X: ourPos.X, Y: ourPos.Y - 2}},
		Head:   ourPos,
		Length: 3,
	}
	
	// Create enemy snakes
	for i := 1; i <= bt.Config.NumEnemies; i++ {
		enemyPos := bt.randomPosition()
		snakes[i] = board.Snake{
			ID:     fmt.Sprintf("enemy%d", i),
			Name:   fmt.Sprintf("Enemy %d", i),
			Health: 100,
			Body:   []board.Coord{enemyPos, {X: enemyPos.X, Y: enemyPos.Y - 1}, {X: enemyPos.X, Y: enemyPos.Y - 2}},
			Head:   enemyPos,
			Length: 3,
		}
	}
	
	// Create food
	food := make([]board.Coord, 3)
	for i := 0; i < 3; i++ {
		food[i] = bt.randomPosition()
	}
	
	return &board.GameState{
		Turn: 0,
		Board: board.Board{
			Height:  bt.Config.BoardHeight,
			Width:   bt.Config.BoardWidth,
			Food:    food,
			Hazards: []board.Coord{},
			Snakes:  snakes,
		},
		You: snakes[0],
	}
}

// randomPosition generates a random board position
func (bt *BatchTester) randomPosition() board.Coord {
	return board.Coord{
		X: bt.rng.Intn(bt.Config.BoardWidth),
		Y: bt.rng.Intn(bt.Config.BoardHeight),
	}
}

// simulateEnemyMoves simulates random enemy moves
func (bt *BatchTester) simulateEnemyMoves(state *board.GameState) *board.GameState {
	newState := state
	
	for _, snake := range state.Board.Snakes {
		if snake.ID == "you" {
			continue
		}
		
		// Get valid moves for enemy
		validMoves := simulation.GetValidMoves(newState, snake.ID)
		if len(validMoves) > 0 {
			// Pick random valid move
			move := validMoves[bt.rng.Intn(len(validMoves))]
			newState = simulation.SimulateMove(newState, snake.ID, move)
		}
	}
	
	return newState
}

// removeDeadSnakes removes snakes that have died
func (bt *BatchTester) removeDeadSnakes(state *board.GameState) *board.GameState {
	aliveSnakes := make([]board.Snake, 0)
	
	for _, snake := range state.Board.Snakes {
		// Check if snake is alive (health > 0 and not out of bounds)
		if snake.Health > 0 && state.Board.IsInBounds(snake.Head) {
			aliveSnakes = append(aliveSnakes, snake)
		}
	}
	
	newState := *state
	newState.Board.Snakes = aliveSnakes
	
	// Update You if it's still alive
	for _, snake := range aliveSnakes {
		if snake.ID == state.You.ID {
			newState.You = snake
			break
		}
	}
	
	return &newState
}

// spawnFood adds random food to the board
func (bt *BatchTester) spawnFood(state *board.GameState) *board.GameState {
	newState := *state
	newFood := bt.randomPosition()
	newState.Board.Food = append(newState.Board.Food, newFood)
	return &newState
}

// isSnakeAlive checks if a snake is still in the game
func isSnakeAlive(state *board.GameState, snakeID string) bool {
	for _, snake := range state.Board.Snakes {
		if snake.ID == snakeID && snake.Health > 0 {
			return true
		}
	}
	return false
}

// categorizeDeathReason determines how the snake died
func categorizeDeathReason(state *board.GameState, turn int) string {
	ourSnake := state.Board.GetSnakeByID("you")
	
	if ourSnake == nil {
		return "eliminated"
	}
	
	// Check starvation
	if ourSnake.Health <= 0 {
		return "starvation"
	}
	
	// Check collision
	if !state.Board.IsInBounds(ourSnake.Head) {
		return "wall-collision"
	}
	
	// Check body collision
	if state.Board.IsOccupied(ourSnake.Head, false) {
		return "body-collision"
	}
	
	// Check head-to-head
	for _, enemy := range state.Board.Snakes {
		if enemy.ID == "you" {
			continue
		}
		if enemy.Head.X == ourSnake.Head.X && enemy.Head.Y == ourSnake.Head.Y {
			if enemy.Length >= ourSnake.Length {
				return "head-to-head-loss"
			}
		}
	}
	
	// Default
	return "unknown"
}

// PrintBatchResults prints a summary of batch test results
func PrintBatchResults(results BatchResults) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║           BATCH TEST RESULTS                               ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	
	fmt.Printf("\nTotal Games:        %d\n", results.TotalGames)
	fmt.Printf("Wins:               %d (%.1f%%)\n", results.Wins, float64(results.Wins)/float64(results.TotalGames)*100)
	fmt.Printf("Losses:             %d (%.1f%%)\n", results.Losses, float64(results.Losses)/float64(results.TotalGames)*100)
	fmt.Printf("Avg Turns:          %.1f\n", results.AvgTurns)
	fmt.Printf("Avg Final Length:   %.1f\n", results.AvgFinalLength)
	fmt.Printf("Avg Food Collected: %.1f\n", results.AvgFoodCollected)
	
	fmt.Println("\n--- Death Reasons ---")
	for reason, count := range results.DeathReasons {
		percentage := float64(count) / float64(results.TotalGames) * 100
		fmt.Printf("  %-20s: %3d (%.1f%%)\n", reason, count, percentage)
	}
	
	fmt.Println("\n════════════════════════════════════════════════════════════")
}
