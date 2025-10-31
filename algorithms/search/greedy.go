package search

import (
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/ErwinsExpertise/go-battleclank/heuristics"
	"github.com/ErwinsExpertise/go-battleclank/policy"
	"math"
)

// Package search provides pluggable search strategies for move selection

// MoveScore represents a scored move
type MoveScore struct {
	Move  string
	Score float64
}

// GreedySearch implements a single-turn greedy heuristic search
type GreedySearch struct {
	SpaceWeight       float64
	HeadCollisionWeight float64
	CenterWeight      float64
	WallPenaltyWeight float64
	CutoffWeight      float64
	MaxDepth          int
	UseAStar          bool
	MaxAStarNodes     int
}

// NewGreedySearch creates a new greedy search with default weights
func NewGreedySearch() *GreedySearch {
	return &GreedySearch{
		SpaceWeight:       100.0,
		HeadCollisionWeight: 500.0,
		CenterWeight:      10.0,
		WallPenaltyWeight: 300.0,
		CutoffWeight:      300.0,
		MaxDepth:          20,
		UseAStar:          true,
		MaxAStarNodes:     200,
	}
}

// FindBestMove evaluates all possible moves and returns the best one
func (g *GreedySearch) FindBestMove(state *board.GameState) string {
	bestMove := board.MoveUp
	bestScore := -math.MaxFloat64
	
	for _, move := range board.AllMoves() {
		score := g.ScoreMove(state, move)
		if score > bestScore {
			bestScore = score
			bestMove = move
		}
	}
	
	return bestMove
}

// ScoreMove evaluates a single move
func (g *GreedySearch) ScoreMove(state *board.GameState, move string) float64 {
	score := 0.0
	myHead := state.You.Head
	nextPos := board.GetNextPosition(myHead, move)
	
	// Check if move is immediately fatal
	if !state.Board.IsInBounds(nextPos) || state.Board.IsOccupied(nextPos, true) {
		return -10000.0
	}
	
	// Calculate aggression score first
	mySpace := heuristics.EvaluateSpace(state, myHead, g.MaxDepth)
	aggression := policy.CalculateAggressionScore(state, mySpace)
	
	// Danger zone evaluation
	dangerZone := heuristics.PredictEnemyDangerZones(state)
	dangerLevel := heuristics.GetDangerLevel(dangerZone, nextPos, state.You.Length)
	score -= dangerLevel
	
	// Space availability
	spaceFactor := heuristics.EvaluateSpace(state, nextPos, g.MaxDepth)
	spaceWeight := g.SpaceWeight
	if hasEnemiesNearby(state, 3) {
		spaceWeight = g.SpaceWeight * 2.0
	}
	score += spaceFactor * spaceWeight
	
	// Food seeking
	foodFactor := heuristics.EvaluateFoodProximity(state, nextPos, g.UseAStar, g.MaxAStarNodes)
	outmatched := policy.IsOutmatched(state, 3)
	foodWeight := policy.GetFoodWeight(state, aggression, outmatched)
	score += foodFactor * foodWeight
	
	// Head collision risk
	headRisk := heuristics.IsHeadToHeadRisky(state, nextPos)
	score -= headRisk * g.HeadCollisionWeight
	
	// Center proximity (early game or when healthy)
	if state.Turn < 50 || (state.You.Health > policy.HealthLow && !outmatched) {
		centerFactor := evaluateCenterProximity(state, nextPos)
		weight := g.CenterWeight
		if state.Turn >= 50 {
			weight = g.CenterWeight * 1.5
		}
		score += centerFactor * weight
	}
	
	// Wall avoidance when enemies exist
	if len(state.Board.Snakes) > 1 {
		wallPenalty := evaluateWallAvoidance(state, nextPos)
		score -= wallPenalty * g.WallPenaltyWeight
	}
	
	// Cutoff detection
	cutoffPenalty := heuristics.DetectCutoff(state, nextPos)
	score -= cutoffPenalty * g.CutoffWeight
	
	// Trap opportunity (if aggressive)
	if policy.ShouldAttemptTrap(aggression) {
		trapScore := heuristics.EvaluateTrapOpportunity(state, nextPos, g.MaxDepth)
		trapWeight := 200.0 * aggression.Score
		score += trapScore * trapWeight
	}
	
	return score
}

// Helper functions

func hasEnemiesNearby(state *board.GameState, radius int) bool {
	myHead := state.You.Head
	
	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}
		
		dist := board.ManhattanDistance(myHead, snake.Head)
		if dist <= radius {
			return true
		}
	}
	
	return false
}

func evaluateCenterProximity(state *board.GameState, pos board.Coord) float64 {
	centerX := state.Board.Width / 2
	centerY := state.Board.Height / 2
	dist := board.ManhattanDistance(pos, board.Coord{X: centerX, Y: centerY})
	maxDist := centerX + centerY
	
	if maxDist == 0 {
		return 1.0
	}
	return 1.0 - (float64(dist) / float64(maxDist))
}

func evaluateWallAvoidance(state *board.GameState, pos board.Coord) float64 {
	distFromLeft := pos.X
	distFromRight := state.Board.Width - 1 - pos.X
	distFromBottom := pos.Y
	distFromTop := state.Board.Height - 1 - pos.Y
	
	minDistToEdge := distFromLeft
	if distFromRight < minDistToEdge {
		minDistToEdge = distFromRight
	}
	if distFromBottom < minDistToEdge {
		minDistToEdge = distFromBottom
	}
	if distFromTop < minDistToEdge {
		minDistToEdge = distFromTop
	}
	
	penalty := 0.0
	switch minDistToEdge {
	case 0:
		penalty = 1.0
	case 1:
		penalty = 0.8
	case 2:
		penalty = 0.5
	case 3:
		penalty = 0.2
	default:
		penalty = 0.0
	}
	
	// Additional corner penalty
	wallsNearby := 0
	if distFromLeft <= 2 {
		wallsNearby++
	}
	if distFromRight <= 2 {
		wallsNearby++
	}
	if distFromBottom <= 2 {
		wallsNearby++
	}
	if distFromTop <= 2 {
		wallsNearby++
	}
	
	if wallsNearby >= 2 {
		penalty += 0.3 * float64(wallsNearby)
	}
	
	return penalty
}
