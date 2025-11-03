package search

import (
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/ErwinsExpertise/go-battleclank/engine/simulation"
	"github.com/ErwinsExpertise/go-battleclank/heuristics"
	"math"
)

// LookaheadSearch implements a multi-turn lookahead search strategy
// Uses a simplified MaxN approach for multi-agent decision making
type LookaheadSearch struct {
	MaxDepth     int
	MaxNodes     int
	baseStrategy *GreedySearch
}

// NewLookaheadSearch creates a new lookahead search
func NewLookaheadSearch(depth int) *LookaheadSearch {
	return &LookaheadSearch{
		MaxDepth:     depth,
		MaxNodes:     1000, // Limit nodes explored to stay under time budget
		baseStrategy: NewGreedySearch(),
	}
}

// FindBestMove evaluates moves using lookahead
func (l *LookaheadSearch) FindBestMove(state *board.GameState) string {
	bestMove := board.MoveUp
	bestScore := -math.MaxFloat64

	// Evaluate each possible move with lookahead
	for _, move := range board.AllMoves() {
		if !simulation.IsMoveValid(state, state.You.ID, move) {
			continue
		}

		score := l.evaluateWithLookahead(state, move, 0)

		if score > bestScore {
			bestScore = score
			bestMove = move
		}
	}

	return bestMove
}

// evaluateWithLookahead recursively evaluates a move with multi-turn lookahead
func (l *LookaheadSearch) evaluateWithLookahead(state *board.GameState, move string, depth int) float64 {
	// Base case: max depth reached or terminal state
	if depth >= l.MaxDepth {
		// Use greedy evaluation at leaf nodes
		return l.baseStrategy.ScoreMove(state, move)
	}

	// Simulate our move
	nextState := simulation.SimulateMove(state, state.You.ID, move)

	// Check if game is over for us
	if !isAlive(nextState, state.You.ID) {
		return -10000.0 - float64(depth)*100 // Penalize earlier deaths more
	}

	// If we survived but it's the last depth, return greedy score
	if depth == l.MaxDepth-1 {
		return l.baseStrategy.ScoreMove(nextState, move)
	}

	// Simplified enemy response prediction
	// Assume enemies will take their best move
	score := l.predictEnemyResponses(nextState, depth+1)

	return score
}

// predictEnemyResponses simulates enemy responses and evaluates the worst-case scenario
func (l *LookaheadSearch) predictEnemyResponses(state *board.GameState, depth int) float64 {
	// Simplified approach: assume enemies move optimally against us
	// In a full MaxN implementation, we'd explore all combinations

	// For now, just evaluate the current state with space and danger metrics
	mySpace := heuristics.EvaluateSpace(state, state.You.Head, 15)

	// Calculate threat from enemies
	dangerZone := heuristics.PredictEnemyDangerZones(state)
	inDanger := heuristics.IsDangerousPosition(dangerZone, state.You.Head)

	score := mySpace * 1000.0

	if inDanger {
		score -= 500.0
	}

	// Bonus for more health
	score += float64(state.You.Health) * 2.0

	// Bonus for length advantage
	avgEnemyLength := getAverageEnemyLength(state)
	if float64(state.You.Length) > avgEnemyLength {
		score += float64(state.You.Length-int(avgEnemyLength)) * 50.0
	}

	return score
}

// isAlive checks if a snake is still alive in the game state
func isAlive(state *board.GameState, snakeID string) bool {
	for _, snake := range state.Board.Snakes {
		if snake.ID == snakeID {
			// Check if health is positive
			return snake.Health > 0
		}
	}
	return false
}

// getAverageEnemyLength calculates the average length of enemy snakes
func getAverageEnemyLength(state *board.GameState) float64 {
	totalLength := 0
	count := 0

	for _, snake := range state.Board.Snakes {
		if snake.ID != state.You.ID {
			totalLength += snake.Length
			count++
		}
	}

	if count == 0 {
		return 0
	}

	return float64(totalLength) / float64(count)
}

// ScoreMove provides a score for a single move (for compatibility)
func (l *LookaheadSearch) ScoreMove(state *board.GameState, move string) float64 {
	return l.evaluateWithLookahead(state, move, 0)
}
