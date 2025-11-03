package search

import (
	"math"

	"github.com/ErwinsExpertise/go-battleclank/config"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/ErwinsExpertise/go-battleclank/heuristics"
	"github.com/ErwinsExpertise/go-battleclank/policy"
)

// Package search provides pluggable search strategies for move selection

const (
	// FatalMoveScore is the score assigned to immediately fatal moves
	// (out of bounds, colliding with snake bodies, etc.)
	FatalMoveScore = -10000.0
)

// MoveScore represents a scored move
type MoveScore struct {
	Move  string
	Score float64
}

// GreedySearch implements a single-turn greedy heuristic search
type GreedySearch struct {
	SpaceWeight            float64
	HeadCollisionWeight    float64
	CenterWeight           float64
	WallPenaltyWeight      float64
	CutoffWeight           float64
	StraightMovementWeight float64
	MaxDepth               int
	UseAStar               bool
	MaxAStarNodes          int
}

// NewGreedySearch creates a new greedy search using config values
func NewGreedySearch() *GreedySearch {
	cfg := config.GetConfig()
	return &GreedySearch{
		SpaceWeight:            cfg.Weights.Space,
		HeadCollisionWeight:    cfg.Weights.HeadCollision,
		CenterWeight:           cfg.Weights.CenterControl,
		WallPenaltyWeight:      cfg.Weights.WallPenalty,
		CutoffWeight:           cfg.Weights.Cutoff,
		StraightMovementWeight: cfg.Weights.StraightMovement,
		MaxDepth:               cfg.Search.MaxDepth,
		UseAStar:               cfg.Search.UseAStar,
		MaxAStarNodes:          cfg.Search.MaxAStarNodes,
	}
}

// FindBestMove evaluates all possible moves and returns the best one
func (g *GreedySearch) FindBestMove(state *board.GameState) string {
	// PRIORITY 1: Check for deterministic cutoff kill opportunity
	// This bypasses all other strategies and scoring
	cutoffMove := heuristics.DetectCutoffKill(state)
	if cutoffMove != "" {
		return cutoffMove
	}

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
		return FatalMoveScore
	}

	// NEW: Straight movement bonus - prefer continuing in current direction
	// This reduces unnecessary turning and creates more efficient paths
	if len(state.You.Body) >= 2 {
		currentDirection := getCurrentDirection(state.You.Head, state.You.Body[1])
		if currentDirection == move {
			// Continuing straight - bonus for movement efficiency
			score += g.StraightMovementWeight
		}
	}

	// NEW: Ratio-based trap detection (matches baseline snake)
	// Use 50% of penalties against random opponents - they won't exploit traps
	_, trapLevel := heuristics.EvaluateSpaceRatio(state, nextPos, g.MaxDepth)
	trapPenalty := heuristics.GetSpaceTrapPenalty(trapLevel) * 0.5
	score -= trapPenalty

	// Calculate space for both current and next position
	mySpace := heuristics.EvaluateSpace(state, myHead, g.MaxDepth)
	nextSpace := heuristics.EvaluateSpace(state, nextPos, g.MaxDepth)

	// CRITICAL: Avoid moves that drastically reduce our space (prevents self-trapping)
	cfg := config.GetConfig()
	if nextSpace < mySpace*cfg.Traps.SpaceReductionRatio60 && mySpace > cfg.Traps.SpaceReductionMinBase {
		// Moving here cuts our space by 60%+ - dangerous!
		score -= cfg.Traps.SpaceReduction60
	} else if nextSpace < mySpace*cfg.Traps.SpaceReductionRatio50 && mySpace > cfg.Traps.SpaceReductionMinBase {
		// Moving here cuts our space by 50%+ - risky
		score -= cfg.Traps.SpaceReduction50
	}

	// NEW: One-move lookahead for dead end detection (matches baseline snake)
	// Use 50% of penalty against random opponents
	deadEndPenalty := heuristics.EvaluateDeadEndAhead(state, nextPos, g.MaxDepth) * 0.5
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

	// EMERGENCY: Wall-side escape logic for head-on collision scenarios
	// When near wall with enemy approaching head-on, prefer turning toward wall for U-turn escape
	wallEscapeBonus := evaluateWallEscapeEmergency(state, myHead, nextPos, move)
	score += wallEscapeBonus

	// Space availability - CRITICAL for survival (use pre-calculated nextSpace)
	spaceFactor := nextSpace
	spaceWeight := g.SpaceWeight * cfg.Weights.SpaceBaseMultiplier

	// Increase space weight when enemies are nearby
	if hasEnemiesNearby(state, 3) {
		spaceWeight = g.SpaceWeight * cfg.Weights.SpaceEnemyMultiplier
	}

	// Bonus for having more space when healthy (allows aggressive play)
	if state.You.Health > policy.HealthLow && spaceFactor > 0.4 {
		spaceWeight *= cfg.Weights.SpaceHealthyMultiplier
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
			foodTrapPenalty := cfg.Traps.FoodTrap
			if state.You.Health < policy.HealthCritical {
				foodTrapPenalty = cfg.Traps.FoodTrapCritical
			} else if state.You.Health < policy.HealthLow {
				foodTrapPenalty = cfg.Traps.FoodTrapLow
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
		survivalBonus := 120.0 // Large bonus for good space
		if state.You.Health < policy.HealthLow {
			survivalBonus = 200.0 // Huge bonus when vulnerable
		}
		score += survivalBonus
	} else if nextSpace > 0.2 {
		// Still reward decent space
		score += 60.0
	}

	// Future options bonus: reward moves with more valid next moves
	validNextMoves := countValidMoves(state, nextPos)
	if validNextMoves >= 3 {
		score += 50.0 // Increased from 30 - having options is very valuable
	} else if validNextMoves == 2 {
		score += 20.0 // Increased from 10
	} else if validNextMoves == 1 {
		score -= 80.0 // Increased penalty - very risky
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
		score += float64(lengthAdvantage) * 30.0 // Moderate bonus for being longer
	}

	// Health maintenance: bonus for maintaining good health
	if state.You.Health > 80 {
		score += 20.0 // Bonus for being healthy
	}

	// Wall approach detection: when heading toward wall, prefer direction with more space
	wallApproachBonus := evaluateWallApproachSpace(state, myHead, nextPos, move, g.MaxDepth)
	score += wallApproachBonus

	return score
}

// Helper functions

// getCurrentDirection determines which direction the snake is currently moving
func getCurrentDirection(head, neck board.Coord) string {
	dx := head.X - neck.X
	dy := head.Y - neck.Y

	if dx > 0 {
		return board.MoveRight
	} else if dx < 0 {
		return board.MoveLeft
	} else if dy > 0 {
		return board.MoveUp
	} else if dy < 0 {
		return board.MoveDown
	}

	// Default if head == neck (shouldn't happen)
	return board.MoveUp
}

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

// evaluateWallApproachSpace detects when heading toward a wall and rewards turning toward more space
// This prevents aggressive moves from running into walls
func evaluateWallApproachSpace(state *board.GameState, currentPos, nextPos board.Coord, move string, maxDepth int) float64 {
	// Detect if we're approaching a wall (within 2 squares)
	distToLeftWall := nextPos.X
	distToRightWall := state.Board.Width - 1 - nextPos.X
	distToBottomWall := nextPos.Y
	distToTopWall := state.Board.Height - 1 - nextPos.Y

	// Find which wall we're closest to
	minDist := distToLeftWall
	approachingWall := ""

	if distToRightWall < minDist {
		minDist = distToRightWall
		approachingWall = "right"
	} else if minDist == distToLeftWall {
		approachingWall = "left"
	}

	if distToBottomWall < minDist {
		minDist = distToBottomWall
		approachingWall = "bottom"
	}

	if distToTopWall < minDist {
		minDist = distToTopWall
		approachingWall = "top"
	}

	// Only activate when very close to wall (within 2 squares)
	if minDist > 2 {
		return 0.0
	}

	// Check if we're moving toward the wall
	movingTowardWall := false
	switch approachingWall {
	case "left":
		movingTowardWall = (move == board.MoveLeft)
	case "right":
		movingTowardWall = (move == board.MoveRight)
	case "bottom":
		movingTowardWall = (move == board.MoveDown)
	case "top":
		movingTowardWall = (move == board.MoveUp)
	}

	// If not moving toward wall, no adjustment needed
	if !movingTowardWall {
		return 0.0
	}

	// We're heading toward a wall! Check perpendicular directions for more space
	// Calculate space in perpendicular directions
	var perp1Move, perp2Move string

	switch move {
	case board.MoveUp, board.MoveDown:
		// Moving vertically, check horizontal space
		perp1Move = board.MoveLeft
		perp2Move = board.MoveRight
	case board.MoveLeft, board.MoveRight:
		// Moving horizontally, check vertical space
		perp1Move = board.MoveUp
		perp2Move = board.MoveDown
	}

	// Calculate space in perpendicular directions
	perp1Pos := board.GetNextPosition(currentPos, perp1Move)
	perp2Pos := board.GetNextPosition(currentPos, perp2Move)

	var perp1Space, perp2Space int
	if state.Board.IsInBounds(perp1Pos) && !state.Board.IsOccupied(perp1Pos, true) {
		perp1Space = heuristics.FloodFill(state, perp1Pos, maxDepth)
	}
	if state.Board.IsInBounds(perp2Pos) && !state.Board.IsOccupied(perp2Pos, true) {
		perp2Space = heuristics.FloodFill(state, perp2Pos, maxDepth)
	}

	// If both perpendicular directions have significantly more space than continuing toward wall,
	// penalize the wall approach
	currentMoveSpace := heuristics.FloodFill(state, nextPos, maxDepth)
	maxPerpSpace := perp1Space
	if perp2Space > maxPerpSpace {
		maxPerpSpace = perp2Space
	}

	// Strong penalty if perpendicular direction has much more space
	if float64(maxPerpSpace) > float64(currentMoveSpace)*1.5 && maxPerpSpace > state.You.Length*2 {
		// The perpendicular direction has significantly more space - penalize wall approach
		spaceDiff := float64(maxPerpSpace - currentMoveSpace)
		return -spaceDiff * 2.0 // Negative bonus (penalty) for approaching wall when better option exists
	}

	return 0.0
}

// evaluateWallEscapeEmergency detects emergency situations near walls where turning toward
// the wall provides a defensive U-turn escape from head-on collision
// This addresses the issue where snake near wall with enemy approaching head-on
// should turn into wall rather than away (which leads to interception)
func evaluateWallEscapeEmergency(state *board.GameState, currentPos, nextPos board.Coord, move string) float64 {
	cfg := config.GetConfig()
	// Only activate when within 1 tile of wall
	distToLeftWall := currentPos.X
	distToRightWall := state.Board.Width - 1 - currentPos.X
	distToBottomWall := currentPos.Y
	distToTopWall := state.Board.Height - 1 - currentPos.Y

	minDistToWall := distToLeftWall
	nearWall := "left"

	if distToRightWall < minDistToWall {
		minDistToWall = distToRightWall
		nearWall = "right"
	}
	if distToBottomWall < minDistToWall {
		minDistToWall = distToBottomWall
		nearWall = "bottom"
	}
	if distToTopWall < minDistToWall {
		minDistToWall = distToTopWall
		nearWall = "top"
	}

	// Only activate when within 1 tile of wall
	if minDistToWall > 1 {
		return 0.0
	}

	// Detect if there's an enemy approaching head-on within 2 tiles
	// "Head-on" means: we're moving in one direction, enemy is moving in opposite/intercepting direction
	hasHeadOnThreat := false
	enemyDistance := 100
	enemyIsLargerOrEqual := false

	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}

		// Check distance to enemy head
		dist := board.ManhattanDistance(currentPos, snake.Head)

		// Enemy must be within detection range to be a threat (not too far, not too close)
		if dist < cfg.EmergencyWallEscape.MinDistance || dist > cfg.EmergencyWallEscape.MaxDistance {
			continue
		}

		// Check if enemy is same size or larger (head-on would be bad for us)
		if snake.Length >= state.You.Length {
			enemyIsLargerOrEqual = true
		}

		// Determine our movement direction
		var ourDirection string
		if len(state.You.Body) >= 2 {
			ourDirection = getCurrentDirection(state.You.Head, state.You.Body[1])
		} else {
			continue
		}

		// Determine enemy's likely direction (from their last movement)
		var enemyDirection string
		if len(snake.Body) >= 2 {
			enemyDirection = getCurrentDirection(snake.Head, snake.Body[1])
		} else {
			continue
		}

		// Check if this is a head-on or intercepting situation
		// Head-on: we're moving toward each other on same axis
		isHeadOn := false

		// Vertical head-on: both on similar X, opposite Y directions
		if (ourDirection == board.MoveUp && enemyDirection == board.MoveDown) ||
			(ourDirection == board.MoveDown && enemyDirection == board.MoveUp) {
			// Check if we're on similar X coordinates (within 3 tiles)
			if abs(currentPos.X-snake.Head.X) <= 3 {
				isHeadOn = true
			}
		}

		// Horizontal head-on: both on similar Y, opposite X directions
		if (ourDirection == board.MoveRight && enemyDirection == board.MoveLeft) ||
			(ourDirection == board.MoveLeft && enemyDirection == board.MoveRight) {
			// Check if we're on similar Y coordinates (within 3 tiles)
			if abs(currentPos.Y-snake.Head.Y) <= 3 {
				isHeadOn = true
			}
		}

		if isHeadOn && enemyIsLargerOrEqual {
			hasHeadOnThreat = true
			if dist < enemyDistance {
				enemyDistance = dist
			}
		}
	}

	// No head-on threat detected
	if !hasHeadOnThreat {
		return 0.0
	}

	// Now check if this move turns us TOWARD the wall (defensive) or AWAY from wall (risky)
	movingTowardWall := false

	switch nearWall {
	case "left":
		movingTowardWall = (move == board.MoveLeft)
	case "right":
		movingTowardWall = (move == board.MoveRight)
	case "bottom":
		movingTowardWall = (move == board.MoveDown)
	case "top":
		movingTowardWall = (move == board.MoveUp)
	}

	// If moving toward wall in this emergency situation, give significant bonus
	// This allows the snake to perform a defensive U-turn maneuver
	if movingTowardWall {
		// Strong bonus to override normal wall avoidance
		// Scaled by how close the enemy is
		emergencyBonus := cfg.EmergencyWallEscape.TurnBonus
		if enemyDistance <= cfg.EmergencyWallEscape.CloseThreshold {
			emergencyBonus = cfg.EmergencyWallEscape.CloseBonus // Even stronger when enemy is very close
		}
		return emergencyBonus
	}

	// If moving away from wall in this emergency, apply penalty
	// This discourages turning into the enemy's interception path
	movingAwayFromWall := false
	switch nearWall {
	case "left":
		movingAwayFromWall = (move == board.MoveRight)
	case "right":
		movingAwayFromWall = (move == board.MoveLeft)
	case "bottom":
		movingAwayFromWall = (move == board.MoveUp)
	case "top":
		movingAwayFromWall = (move == board.MoveDown)
	}

	if movingAwayFromWall {
		// Penalty for turning away from wall toward potential interception
		return -cfg.EmergencyWallEscape.AwayPenalty
	}

	// Perpendicular moves are neutral in this scenario
	return 0.0
}

// abs returns absolute value of an integer
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
