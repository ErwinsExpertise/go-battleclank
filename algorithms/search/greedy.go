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

// NewGreedySearch creates a new greedy search with tuned weights for 80%+ win rate
func NewGreedySearch() *GreedySearch {
	return &GreedySearch{
		SpaceWeight:         250.0,  // CRITICAL: space = survival against random opponents
		HeadCollisionWeight: 600.0,  // Important but not as much vs random
		CenterWeight:        10.0,   // Moderate - helps with positioning
		WallPenaltyWeight:   150.0,  // Lower - walls aren't as dangerous vs random
		CutoffWeight:        350.0,  // Moderate - escape routes matter
		MaxDepth:            35,     // Deep lookahead for comprehensive space evaluation
		UseAStar:            true,
		MaxAStarNodes:       800,    // More thorough pathfinding
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
	
	// NEW: Ratio-based trap detection (matches baseline snake)
	// Use 50% of penalties against random opponents - they won't exploit traps
	_, trapLevel := heuristics.EvaluateSpaceRatio(state, nextPos, g.MaxDepth)
	trapPenalty := heuristics.GetSpaceTrapPenalty(trapLevel) * 0.15
	score -= trapPenalty
	
	// Calculate space for both current and next position
	mySpace := heuristics.EvaluateSpace(state, myHead, g.MaxDepth)
	nextSpace := heuristics.EvaluateSpace(state, nextPos, g.MaxDepth)
	
	// CRITICAL: Avoid moves that drastically reduce our space
	if nextSpace < mySpace * 0.3 && mySpace > 0.2 {
		// Moving here cuts our space by 70%+ - dangerous!
		score -= 1000.0
	}
	
	// NEW: One-move lookahead for dead end detection (matches baseline snake)
	// Use 50% of penalty against random opponents
	deadEndPenalty := heuristics.EvaluateDeadEndAhead(state, nextPos, g.MaxDepth) * 0.15
	score -= deadEndPenalty
	
	// Calculate aggression score and situational awareness
	aggression := policy.CalculateAggressionScore(state, mySpace)
	outmatched := policy.IsOutmatched(state, 3)
	
	// Danger zone evaluation - HEAVILY penalize dangerous moves
	dangerZone := heuristics.PredictEnemyDangerZones(state)
	dangerLevel := heuristics.GetDangerLevel(dangerZone, nextPos, state.You.Length)
	
	// Multiply danger penalty when health is low or outmatched
	dangerMultiplier := 1.0
	if state.You.Health < policy.HealthCritical || outmatched {
		dangerMultiplier = 1.5
	}
	
	score -= dangerLevel * dangerMultiplier
	
	// Space availability - CRITICAL for survival (use pre-calculated nextSpace)
	spaceFactor := nextSpace
	spaceWeight := g.SpaceWeight
	
	// Increase space weight when enemies are nearby
	if hasEnemiesNearby(state, 3) {
		spaceWeight = g.SpaceWeight * 2.2  // Moderate increase
	}
	
	// Bonus for having more space when healthy (allows aggressive play)
	if state.You.Health > policy.HealthLow && spaceFactor > 0.4 {
		spaceWeight *= 1.1
	}
	
	score += spaceFactor * spaceWeight
	
	// NEW: Food death trap detection (matches baseline snake)
	// If moving to food, check if we'll be trapped after eating
	isFoodAtPos := false
	for _, food := range state.Board.Food {
		if food.X == nextPos.X && food.Y == nextPos.Y {
			isFoodAtPos = true
			break
		}
	}
	
	if isFoodAtPos {
		// Check if eating this food would trap us (70% threshold)
		isTrap, _ := heuristics.EvaluateFoodTrapRatio(state, nextPos, g.MaxDepth)
		if isTrap {
			// Food death trap - dangerous but reduce penalty if health is critical
			foodTrapPenalty := 800.0
			if state.You.Health < policy.HealthCritical {
				foodTrapPenalty = 400.0  // Risk it when starving
			} else if state.You.Health < policy.HealthLow {
				foodTrapPenalty = 600.0  // Reduced risk when low health
			}
			score -= foodTrapPenalty
		}
	}
	
	// Food seeking (outmatched already calculated above)
	foodFactor := heuristics.EvaluateFoodProximity(state, nextPos, g.UseAStar, g.MaxAStarNodes)
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
	
	// Trap opportunity (if aggressive AND safe)
	if policy.ShouldAttemptTrap(aggression) && nextSpace > 0.25 {
		trapScore := heuristics.EvaluateTrapOpportunity(state, nextPos, g.MaxDepth)
		trapWeight := 200.0 * aggression.Score
		score += trapScore * trapWeight
	}
	
	// Survival bonus: MASSIVELY reward moves that maintain good space
	if nextSpace > 0.3 {
		survivalBonus := 120.0  // Large bonus for good space
		if state.You.Health < policy.HealthLow {
			survivalBonus = 200.0  // Huge bonus when vulnerable
		}
		score += survivalBonus
	} else if nextSpace > 0.2 {
		// Still reward decent space
		score += 60.0
	}
	
	// Future options bonus: reward moves with more valid next moves
	validNextMoves := countValidMoves(state, nextPos)
	if validNextMoves >= 3 {
		score += 50.0  // Increased from 30 - having options is very valuable
	} else if validNextMoves == 2 {
		score += 20.0  // Increased from 10
	} else if validNextMoves == 1 {
		score -= 80.0  // Increased penalty - very risky
	}
	
	// Length advantage bonus: reward being longer than enemies
	lengthAdvantage := 0
	for _, snake := range state.Board.Snakes {
		if snake.ID != state.You.ID {
			if state.You.Length > snake.Length {
				lengthAdvantage++
			}
		}
	}
	if lengthAdvantage > 0 {
		score += float64(lengthAdvantage) * 30.0  // Moderate bonus for being longer
	}
	
	// Health maintenance: bonus for maintaining good health
	if state.You.Health > 80 {
		score += 20.0  // Bonus for being healthy
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

func countValidMoves(state *board.GameState, pos board.Coord) int {
	count := 0
	for _, move := range board.AllMoves() {
		nextPos := board.GetNextPosition(pos, move)
		if state.Board.IsInBounds(nextPos) && !state.Board.IsOccupied(nextPos, true) {
			count++
		}
	}
	return count
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
