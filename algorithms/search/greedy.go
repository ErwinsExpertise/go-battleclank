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

// NewGreedySearch creates a new greedy search TUNED TO MATCH BASELINE
// Weights carefully aligned with baseline Rust snake's scoring.rs
func NewGreedySearch() *GreedySearch {
	return &GreedySearch{
		SpaceWeight:         5.0,    // Baseline: 5 per open square
		HeadCollisionWeight: 500.0,  // Baseline: 500/-500 for win/lose head-to-head
		CenterWeight:        2.0,    // Baseline: 2 for center control
		WallPenaltyWeight:   5.0,    // Baseline: -5 near wall penalty
		CutoffWeight:        200.0,  // Baseline: -200 for dead end ahead
		MaxDepth:            121,    // Baseline: caps at 11x11 = 121 squares
		UseAStar:            true,
		MaxAStarNodes:       400,    // Keep A* for food seeking
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
	
	// Ratio-based trap detection - MATCHED TO BASELINE
	// Use FULL penalties like baseline does (-250, -450, -600)
	_, trapLevel := heuristics.EvaluateSpaceRatio(state, nextPos, g.MaxDepth)
	trapPenalty := heuristics.GetSpaceTrapPenalty(trapLevel)
	score -= trapPenalty
	
	// Calculate space for both current and next position
	mySpace := heuristics.EvaluateSpace(state, myHead, g.MaxDepth)
	nextSpace := heuristics.EvaluateSpace(state, nextPos, g.MaxDepth)
	
	// CRITICAL: Avoid moves that drastically reduce our space
	if nextSpace < mySpace * 0.3 && mySpace > 0.2 {
		// Moving here cuts our space by 70%+ - dangerous!
		score -= 1000.0
	}
	
	// One-move lookahead - MATCHED TO BASELINE
	// Use FULL penalty like baseline: -200 for dead end ahead
	deadEndPenalty := heuristics.EvaluateDeadEndAhead(state, nextPos, g.MaxDepth)
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
		// MATCHED TO BASELINE: -800 food death trap penalty
		isTrap, _ := heuristics.EvaluateFoodTrapRatio(state, nextPos, g.MaxDepth)
		if isTrap {
			// Food death trap - baseline uses flat -800, we reduce when critical
			foodTrapPenalty := 800.0
			if state.You.Health < policy.HealthCritical {
				foodTrapPenalty = 400.0  // Risk it when starving
			} else if state.You.Health < policy.HealthLow {
				foodTrapPenalty = 600.0  // Reduced risk when low health
			}
			score -= foodTrapPenalty
		}
	}
	
	// AGGRESSIVE PURSUIT - MATCHED TO BASELINE
	// Baseline: pursuit bonus when longer (100/50/25/10 at dist 2/3/4/5)
	pursuitBonus := evaluateAggressivePursuit(state, nextPos)
	score += pursuitBonus
	
	// AGGRESSIVE TRAPPING: Additional trap detection
	trapBonus := evaluateAggressiveTrapping(state, nextPos, mySpace, aggression.Score)
	score += trapBonus * 400.0  // High weight on trapping opportunities
	
	// Food seeking (outmatched already calculated above)
	foodFactor := heuristics.EvaluateFoodProximity(state, nextPos, g.UseAStar, g.MaxAStarNodes)
	foodWeight := policy.GetFoodWeight(state, aggression, outmatched)
	
	// INCREMENTAL IMPROVEMENT 1: Simple health-based urgency (lightweight)
	// Just a multiplier based on health - no complex logic
	if state.You.Health < policy.HealthCritical {
		foodWeight *= 1.8  // Critical health: big boost (was 2.0, reduced for balance)
	} else if state.You.Health < policy.HealthLow {
		foodWeight *= 1.4  // Low health: moderate boost (was 1.5, reduced)
	}
	
	// Be more aggressive with food when we have advantage
	if aggression.Score > 0.6 {
		foodWeight *= 1.3  // 30% boost when aggressive
	}
	score += foodFactor * foodWeight
	
	// Head collision risk - reduced when aggressive
	headRisk := heuristics.IsHeadToHeadRisky(state, nextPos)
	headRiskMultiplier := 1.0
	if aggression.Score > 0.6 {
		headRiskMultiplier = 0.7  // 30% less cautious when aggressive
	}
	
	// INCREMENTAL IMPROVEMENT 2: Very conservative late-game adjustment
	// Only in late game (120+ turns) and only slight adjustment
	if state.Turn > 120 && state.You.Health > policy.HealthLow {
		enemiesAlive := len(state.Board.Snakes) - 1
		// If we're in final 1v1 and have length advantage, be slightly more cautious
		if enemiesAlive == 1 {
			avgLen := 0
			for _, snake := range state.Board.Snakes {
				if snake.ID != state.You.ID {
					avgLen = snake.Length
					break
				}
			}
			if state.You.Length > avgLen {
				headRiskMultiplier *= 1.10  // Just 10% more cautious (was 15%, reduced)
			}
		}
	}
	
	score -= headRisk * g.HeadCollisionWeight * headRiskMultiplier
	
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

// evaluateAggressiveTrapping calculates bonus for moves that trap the enemy
// Returns 0.0-1.0 score bonus for trapping opportunities
func evaluateAggressiveTrapping(state *board.GameState, nextPos board.Coord, mySpace float64, aggression float64) float64 {
// Only trap when aggressive (good health, length advantage)
if aggression < 0.6 {
return 0.0
}

trapScore := 0.0

// Check each enemy snake
for _, enemy := range state.Board.Snakes {
if enemy.ID == state.You.ID {
continue
}

// Only trap if we're longer (can win head-to-head)
if enemy.Length >= state.You.Length {
continue
}

// Calculate distance to enemy head
distToEnemy := manhattanDistance(nextPos, enemy.Head)

// Bonus for getting close to smaller enemies
if distToEnemy <= 3 {
trapScore += (4.0 - float64(distToEnemy)) / 4.0 * 0.3
}

// Check if this move reduces enemy's space significantly
enemySpace := heuristics.FloodFillForSnake(state, enemy.ID, enemy.Head, 40)

// Simulate board after our move
simState := simulateMoveForTrapping(state, nextPos)
enemySpaceAfter := heuristics.FloodFillForSnake(simState, enemy.ID, enemy.Head, 40)

// If enemy loses significant space, big bonus
spaceReduction := float64(enemySpace - enemySpaceAfter)
if spaceReduction > 0 {
reductionRatio := spaceReduction / float64(enemySpace)

if reductionRatio > 0.2 {  // Enemy loses 20%+ space
trapScore += reductionRatio * 0.5
}

// Extra bonus if enemy gets trapped (low space relative to body)
enemyRatio := float64(enemySpaceAfter) / float64(enemy.Length)
if enemyRatio < 0.6 {  // Enemy trapped
trapScore += 0.3
}
}
}

return trapScore
}

// simulateMoveForTrapping creates a simulated state after our move
func simulateMoveForTrapping(state *board.GameState, ourNewHead board.Coord) *board.GameState {
simState := &board.GameState{
Board: board.Board{
Width:  state.Board.Width,
Height: state.Board.Height,
Food:   state.Board.Food,
Snakes: make([]board.Snake, len(state.Board.Snakes)),
},
You:  state.You,
Turn: state.Turn,
}

// Copy snakes and update our position
for i, snake := range state.Board.Snakes {
if snake.ID == state.You.ID {
// Our snake with new head position
newBody := make([]board.Coord, len(snake.Body))
newBody[0] = ourNewHead
copy(newBody[1:], snake.Body[:len(snake.Body)-1])

simState.Board.Snakes[i] = board.Snake{
ID:     snake.ID,
Name:   snake.Name,
Health: snake.Health,
Body:   newBody,
Head:   ourNewHead,
Length: snake.Length,
}
simState.You = simState.Board.Snakes[i]
} else {
// Other snakes stay the same
simState.Board.Snakes[i] = snake
}
}

return simState
}

func manhattanDistance(a, b board.Coord) int {
dx := a.X - b.X
if dx < 0 {
dx = -dx
}
dy := a.Y - b.Y
if dy < 0 {
dy = -dy
}
return dx + dy
}

// evaluateTerritorialAdvantage rewards moves that give us more space than enemies
func evaluateTerritorialAdvantage(state *board.GameState, nextPos board.Coord, mySpace float64) float64 {
if len(state.Board.Snakes) <= 1 {
return 0.0
}

bonus := 0.0

// Calculate our space advantage
totalEnemySpace := 0
enemyCount := 0

for _, enemy := range state.Board.Snakes {
if enemy.ID == state.You.ID {
continue
}

enemySpace := heuristics.FloodFillForSnake(state, enemy.ID, enemy.Head, 35)
totalEnemySpace += enemySpace
enemyCount++
}

if enemyCount == 0 {
return 0.0
}

avgEnemySpace := float64(totalEnemySpace) / float64(enemyCount)
mySpaceCount := heuristics.FloodFill(state, nextPos, 35)

// Bonus if we have more space than average enemy
if float64(mySpaceCount) > avgEnemySpace {
spaceAdvantage := (float64(mySpaceCount) - avgEnemySpace) / avgEnemySpace
bonus += spaceAdvantage * 0.5
}

// Extra bonus for dominating (2x+ their space)
if float64(mySpaceCount) >= avgEnemySpace * 2.0 {
bonus += 0.3
}

return bonus
}

// evaluateAggressivePursuit matches baseline's pursuit logic (lines 202-217)
// Move toward enemy heads if we outsize them, scaled inversely with distance
func evaluateAggressivePursuit(state *board.GameState, nextPos board.Coord) float64 {
bonus := 0.0

for _, enemy := range state.Board.Snakes {
if enemy.ID == state.You.ID {
continue
}

// Only pursue if we're longer
if state.You.Length <= enemy.Length {
continue
}

dist := manhattanDistance(nextPos, enemy.Head)

// Baseline pursuit bonuses at distances 2-5
pursuitScore := 0.0
switch dist {
case 2:
pursuitScore = 100.0  // Almost in range - very good
case 3:
pursuitScore = 50.0   // Closing in
case 4:
pursuitScore = 25.0   // Still relevant
case 5:
pursuitScore = 10.0   // On radar
}

bonus += pursuitScore
}

return bonus
}

// evaluateAggressivePursuit matches baseline's pursuit logic
