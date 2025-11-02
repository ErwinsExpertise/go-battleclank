package gpu

import (
	"fmt"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/ErwinsExpertise/go-battleclank/engine/simulation"
	"math"
	"math/rand"
	"sync"
	"time"
)

// MCTSBatchGPU runs multiple MCTS simulations in parallel on GPU (or CPU fallback)
type MCTSBatchGPU struct {
	batchSize    int
	maxDepth     int
	explorationC float64
	rng          *rand.Rand
}

// MCTSResult holds the results for a single move option
type MCTSResult struct {
	Move   string
	Visits int
	Wins   float64
	Score  float64
}

// NewMCTSBatchGPU creates a new batch MCTS runner
func NewMCTSBatchGPU(batchSize, maxDepth int) *MCTSBatchGPU {
	return &MCTSBatchGPU{
		batchSize:    batchSize,
		maxDepth:     maxDepth,
		explorationC: math.Sqrt(2),
		rng:          rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// SimulateBatch runs multiple MCTS simulations in parallel
// When GPU is available, this will leverage parallel execution on the GPU
func (m *MCTSBatchGPU) SimulateBatch(state *board.GameState, iterations int) (string, error) {
	if !IsAvailable() {
		return "", fmt.Errorf("GPU not available")
	}
	
	// Get valid moves
	validMoves := simulation.GetValidMoves(state, state.You.ID)
	if len(validMoves) == 0 {
		return board.MoveUp, fmt.Errorf("no valid moves")
	}
	
	// Initialize results for each move
	results := make([]MCTSResult, len(validMoves))
	for i, move := range validMoves {
		results[i] = MCTSResult{Move: move}
	}
	
	// Run simulations in batches
	// Future: This will be executed on GPU for true parallel processing
	numBatches := (iterations + m.batchSize - 1) / m.batchSize
	
	for batch := 0; batch < numBatches; batch++ {
		currentBatchSize := m.batchSize
		if batch == numBatches-1 {
			remaining := iterations - batch*m.batchSize
			if remaining < currentBatchSize {
				currentBatchSize = remaining
			}
		}
		
		// Run batch simulations (CPU fallback for now)
		m.runBatchCPU(state, validMoves, currentBatchSize, results)
	}
	
	// Select best move based on visit count
	return selectBestMove(results), nil
}

// runBatchCPU executes a batch of simulations on CPU (fallback)
func (m *MCTSBatchGPU) runBatchCPU(state *board.GameState, moves []string, batchSize int, results []MCTSResult) {
	var wg sync.WaitGroup
	
	// Distribute simulations across moves
	for i := range moves {
		wg.Add(1)
		go func(moveIdx int) {
			defer wg.Done()
			
			// Run simulations for this move
			for sim := 0; sim < batchSize/len(moves)+1; sim++ {
				// Simulate the move
				newState := simulation.SimulateMove(state, state.You.ID, moves[moveIdx])
				
				// Run random playout
				reward := m.simulatePlayout(newState)
				
				// Update results
				results[moveIdx].Visits++
				results[moveIdx].Wins += reward
			}
		}(i)
	}
	
	wg.Wait()
}

// simulatePlayout performs a random playout from a given state
func (m *MCTSBatchGPU) simulatePlayout(state *board.GameState) float64 {
	currentState := state
	depth := 0
	
	for depth < m.maxDepth {
		// Check if our snake is still alive
		ourSnake := currentState.Board.GetSnakeByID(currentState.You.ID)
		if ourSnake == nil || ourSnake.Health <= 0 {
			return 0.0 // Lost
		}
		
		// Check if we won
		aliveCount := 0
		for _, snake := range currentState.Board.Snakes {
			if snake.Health > 0 {
				aliveCount++
			}
		}
		if aliveCount == 1 {
			return 1.0 // Won
		}
		
		// Get valid moves and pick random one
		validMoves := simulation.GetValidMoves(currentState, currentState.You.ID)
		if len(validMoves) == 0 {
			return 0.0
		}
		
		move := validMoves[m.rng.Intn(len(validMoves))]
		currentState = simulation.SimulateMove(currentState, currentState.You.ID, move)
		
		depth++
	}
	
	// Evaluate final state heuristically
	return evaluateState(currentState)
}

// evaluateState provides a heuristic evaluation of a game state
func evaluateState(state *board.GameState) float64 {
	ourSnake := state.Board.GetSnakeByID(state.You.ID)
	if ourSnake == nil || ourSnake.Health <= 0 {
		return 0.0
	}
	
	// Simple heuristic: health and relative size
	healthScore := float64(ourSnake.Health) / 100.0
	
	// Count enemy snakes and compare sizes
	numEnemies := 0
	largerEnemies := 0
	for _, snake := range state.Board.Snakes {
		if snake.ID != state.You.ID && snake.Health > 0 {
			numEnemies++
			if snake.Length > ourSnake.Length {
				largerEnemies++
			}
		}
	}
	
	sizeScore := 0.5
	if numEnemies > 0 {
		sizeScore = 1.0 - float64(largerEnemies)/float64(numEnemies)
	}
	
	return (healthScore*0.3 + sizeScore*0.7)
}

// selectBestMove returns the move with the highest visit count
func selectBestMove(results []MCTSResult) string {
	if len(results) == 0 {
		return board.MoveUp
	}
	
	bestIdx := 0
	bestVisits := results[0].Visits
	
	for i := 1; i < len(results); i++ {
		if results[i].Visits > bestVisits {
			bestIdx = i
			bestVisits = results[i].Visits
		}
	}
	
	return results[bestIdx].Move
}
