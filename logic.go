package main

import (
	"container/heap"
	"log"
	"math"
)

// This package implements a STATELESS Battlesnake AI.
// All decision-making functions are pure functions that depend only on the
// current GameState parameter, with no persistent memory between turns.
// This ensures:
// - Updated decisions based on current board state
// - Easy debugging and testing
// - Simple horizontal scaling
// - No accumulation of stale state or bugs
// See: https://medium.com/asymptoticlabs/battlesnake-post-mortem-a5917f9a3428

const (
	// Move directions
	MoveUp    = "up"
	MoveDown  = "down"
	MoveLeft  = "left"
	MoveRight = "right"

	// Health constants
	HealthCritical = 30
	HealthLow      = 50
	MaxHealth      = 100

	// A* algorithm constants
	// MaxAStarNodes limits the number of nodes explored in A* search to prevent timeouts.
	// Tuning guidance:
	// - 11x11 board: 200 nodes (default) - balances accuracy and performance
	// - 7x7 board: 100 nodes - smaller board requires fewer nodes
	// - 19x19 board: 300-400 nodes - larger board may need more exploration
	// Average search explores 50-150 nodes, so 200 provides good headroom.
	MaxAStarNodes = 200

	// Food danger detection constants
	// FoodDangerRadius defines how close (in Manhattan distance) food must be to an enemy
	// snake to be considered dangerous. A radius of 2 provides a safety buffer while not
	// being overly conservative.
	// Tuning guidance:
	// - Radius 1: Very aggressive, only avoids food directly adjacent to enemies
	// - Radius 2: Balanced (default) - avoids traps while still seeking most food
	// - Radius 3: Conservative - may miss food opportunities but maximizes safety
	FoodDangerRadius = 2

	// FoodDangerPenalty is the multiplier applied to food scores when food is dangerous.
	// A penalty of 0.1 means dangerous food is 90% less attractive.
	FoodDangerPenalty = 0.1

	// EnemyProximityRadius defines how close (in Manhattan distance) an enemy snake
	// must be to our head to be considered "nearby". When enemies are within this
	// radius, tail-chasing behavior is disabled to avoid circling near threats.
	// Tuning guidance:
	// - Radius 2: Very cautious - disables circling even with distant enemies
	// - Radius 3: Balanced (default) - reasonable safety zone
	// - Radius 4-5: More relaxed - only disables circling when enemies are very close
	EnemyProximityRadius = 3

	// SpaceBufferRadius defines how close we consider enemy snakes when evaluating space.
	// When flood filling, we treat positions within this radius of enemy heads as blocked
	// to account for the fact that enemies can move and cut off our escape routes.
	// Tuning guidance:
	// - Radius 1: Aggressive - only avoids immediate enemy positions
	// - Radius 2: Balanced (default) - accounts for one enemy move ahead
	// - Radius 3: Conservative - very cautious, may over-restrict movement
	SpaceBufferRadius = 2

	// Trap detection constants
	// TrapSpaceThreshold is the minimum percentage of space an enemy must lose for us
	// to consider a move as a potential trap. Value of 0.15 means enemy loses 15%+ space.
	TrapSpaceThreshold = 0.15

	// TrapSafetyMargin is the minimum space advantage we need to maintain when trapping.
	// Value of 1.2 means we need 20% more reachable space than the enemy we're trapping.
	TrapSafetyMargin = 1.2

	// Aggression scoring thresholds
	// AggressionLengthAdvantage is the length advantage needed for aggressive behavior
	AggressionLengthAdvantage = 2

	// AggressionHealthThreshold is the minimum health needed for aggressive behavior
	AggressionHealthThreshold = 60
)

// info returns metadata about the battlesnake
func info() BattlesnakeInfoResponse {
	log.Println("INFO")
	// Use build version if available, otherwise default
	buildVersion := version
	if buildVersion == "dev" {
		buildVersion = "1.0.0"
	}
	return BattlesnakeInfoResponse{
		APIVersion: "1",
		Author:     "go-battleclank",
		Color:      "#FF6B35",
		Head:       "trans-rights-scarf",
		Tail:       "bolt",
		Version:    buildVersion,
	}
}

// start is called when the game begins
// This function is intentionally stateless - it does not initialize or store
// any game state. All decisions are made fresh on each move based solely on
// the GameState parameter provided by the Battlesnake API.
func start(state GameState) {
	log.Printf("GAME START: %s\n", state.Game.ID)
}

// end is called when the game finishes
// This function is intentionally stateless - it does not clean up or persist
// any game state. This ensures the server remains simple and scalable.
func end(state GameState) {
	log.Printf("GAME OVER: %s\n", state.Game.ID)
}

// move is the main decision-making function called on every turn
// This function is STATELESS and makes decisions based purely on the current
// GameState parameter, ensuring fresh, updated decisions every turn without
// any dependency on previous game history or persistent state.
func move(state GameState) BattlesnakeMoveResponse {
	possibleMoves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}

	// Score each possible move
	moveScores := make(map[string]float64)
	for _, move := range possibleMoves {
		moveScores[move] = scoreMove(state, move)
	}

	// Find the best move
	// Iterate in a fixed order to ensure deterministic behavior.
	// When multiple moves have the same score, we'll consistently pick the first one
	// in the possibleMoves array (up, down, left, right), ensuring stateless behavior.
	bestMove := MoveUp
	bestScore := -math.MaxFloat64
	for _, move := range possibleMoves {
		score := moveScores[move]
		if score > bestScore {
			bestScore = score
			bestMove = move
		}
	}

	log.Printf("MOVE %d: %s (score: %.2f)\n", state.Turn, bestMove, bestScore)
	return BattlesnakeMoveResponse{
		Move: bestMove,
	}
}

// scoreMove evaluates how good a move is using multiple heuristics
func scoreMove(state GameState, move string) float64 {
	score := 0.0
	myHead := state.You.Head
	nextPos := getNextPosition(myHead, move)

	// Check if move is immediately fatal
	if isImmediatelyFatal(state, nextPos) {
		return -10000.0
	}

	// CRITICAL: Check for food death traps (from baseline analysis)
	// This must be checked before other evaluations as it prevents fatal mistakes
	foodDeathTrapPenalty := evaluateFoodDeathTrap(state, nextPos, nextPos)
	if foodDeathTrapPenalty > 0 {
		score -= foodDeathTrapPenalty
	}

	// CRITICAL: Check for ratio-based trap detection (from baseline analysis)
	// Applies graduated penalties based on space-to-body-length ratio
	trapPenalty := evaluateTrapPenalty(state, nextPos)
	score -= trapPenalty

	// HIGH PRIORITY: One-move lookahead to detect dead-ends
	// Checks if this move leads to limited future options
	lookaheadPenalty := evaluateOneMoveAhead(state, move)
	score -= lookaheadPenalty

	// OLD: Disabled multi-turn lookahead in favor of new ratio-based trap detection
	// The new trap detection (ratio-based + one-move lookahead) is more accurate
	// and less computationally expensive
	// if !simulateMove(state, move, 1) {
	// 	score -= 3000.0
	// }

	// NEW: Check if we're moving into enemy danger zones
	// Predict where enemies can move next turn
	enemyMoves := predictEnemyMoves(state)
	if enemies, inDanger := enemyMoves[nextPos]; inDanger {
		// We're moving into a position an enemy could also move to
		for _, enemy := range enemies {
			if enemy.Length > state.You.Length+1 {
				// Enemy is significantly larger - very dangerous
				score -= 700.0 // Reduced from 800 to be less conservative
			} else if enemy.Length == state.You.Length || enemy.Length == state.You.Length+1 {
				// Enemy is same size or slightly larger - risky but contestable
				score -= 400.0 // Reduced from 800 to allow contesting
			} else {
				// Enemy is smaller - we have advantage, minimal penalty
				score -= 100.0 // Reduced from 200 to be more aggressive
			}
		}
	}

	// Space availability (flood fill)
	// Increase weight when enemies are nearby (more important to maintain space)
	spaceFactor := evaluateSpace(state, nextPos)
	spaceWeight := 100.0
	if hasEnemiesNearby(state) {
		spaceWeight = 200.0 // Double weight when enemies are close
	}
	score += spaceFactor * spaceWeight

	// Food seeking - always seek food to avoid starvation and circular behavior
	// Weight increases as health decreases
	// Slightly reduced when we're significantly smaller than nearby enemies (defensive play)
	foodFactor := evaluateFoodProximity(state, nextPos)
	foodWeight := 0.0
	
	// Check if we're outmatched by nearby enemies
	outmatched := isOutmatchedByNearbyEnemies(state)
	
	if state.You.Health < HealthCritical {
		// Critical health: MAXIMUM food seeking - survival is paramount
		// Even when outmatched, we must eat or die
		foodWeight = 400.0
		if outmatched {
			foodWeight = 350.0 // Only slight reduction when outmatched
		}
	} else if state.You.Health < HealthLow {
		// Low health: strong food seeking
		foodWeight = 250.0
		if outmatched {
			foodWeight = 180.0 // Moderate reduction when outmatched
		}
	} else {
		// Healthy: ALWAYS aggressively seek food to maintain dominance and growth
		// Increased from 100 to 150 to eliminate circling behavior
		// Snake should always be hunting, never standing still
		foodWeight = 150.0
		if outmatched {
			foodWeight = 90.0 // Still seek food actively when outmatched
		}
	}
	score += foodFactor * foodWeight

	// Avoid enemy snakes' heads (they might kill us in head-to-head)
	headCollisionRisk := evaluateHeadCollisionRisk(state, nextPos)
	score -= headCollisionRisk * 500.0

	// Prefer center positions early game and when healthy
	// Center control is key to aggressive play and territory dominance
	if state.Turn < 50 {
		centerFactor := evaluateCenterProximity(state, nextPos)
		score += centerFactor * 10.0
	} else if state.You.Health > HealthLow && !outmatched {
		// When healthy and not outmatched, maintain center control
		centerFactor := evaluateCenterProximity(state, nextPos)
		score += centerFactor * 15.0
	}

	// Tail following is now DISABLED to prevent circular behavior
	// The snake should always be hunting for food or prey, never standing still
	// Only use tail as fallback when no other options exist (very low weight)
	// Note: Using HealthLow (50) as threshold to ensure we're healthy enough for this fallback
	if state.You.Health > HealthLow && !hasEnemiesNearby(state) && len(state.Board.Food) == 0 {
		// Only follow tail when no food exists (rare case)
		tailFactor := evaluateTailProximity(state, nextPos)
		score += tailFactor * 5.0 // Reduced from 50 to 5 - minimal weight
	}

	// Penalize corner/edge positions when enemies exist (not just nearby)
	// This prevents getting squeezed into corners with limited escape routes
	// Reduced from 400 to 300 to be less wall-averse and allow contesting food near edges
	if hasAnyEnemies(state) {
		wallPenalty := evaluateWallAvoidance(state, nextPos)
		score -= wallPenalty * 300.0
	}

	// NEW: Detect cutoff/boxing-in scenarios
	// Heavy penalty for positions with limited escape routes
	cutoffPenalty := detectCutoff(state, nextPos)
	score -= cutoffPenalty * 300.0

	// NEW: Anti-chasing behavior
	// If being chased, prefer moves that break the pattern or use our tail space
	if isBeingChased(state) {
		// Slightly prefer moves toward our own body area where enemy can't easily follow
		// This creates unpredictable movement
		myTail := state.You.Body[len(state.You.Body)-1]
		distToTail := manhattanDistance(nextPos, myTail)
		// Small bonus for moving toward areas we control
		if distToTail <= 2 {
			score += 20.0
		}
	}

	// NEW: Calculate aggression score to determine offensive vs defensive play
	aggressionScore := calculateAggressionScore(state)
	
	// NEW: Evaluate trap opportunities when we're in aggressive mode
	// Only attempt traps when we have the advantage (high aggression score)
	if aggressionScore > 0.6 {
		trapScore := evaluateTrapOpportunity(state, nextPos)
		// Weight trap opportunities based on aggression level
		// Higher aggression = more willing to pursue traps
		trapWeight := 200.0 * aggressionScore
		score += trapScore * trapWeight
	}

	return score
}

// isOutmatchedByNearbyEnemies checks if there are nearby enemies significantly larger than us
// Returns true if we should play defensively
func isOutmatchedByNearbyEnemies(state GameState) bool {
	myHead := state.You.Head
	myLength := state.You.Length
	
	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}
		
		// Check if enemy is nearby (within proximity radius)
		dist := manhattanDistance(myHead, snake.Head)
		if dist <= EnemyProximityRadius {
			// Enemy is nearby - check if they're significantly larger
			// Increased threshold from +2 to +4 to be less defensive
			if snake.Length > myLength+4 {
				return true
			}
		}
	}
	
	return false
}

// isImmediatelyFatal checks if a position results in immediate death
func isImmediatelyFatal(state GameState, pos Coord) bool {
	// Out of bounds
	if pos.X < 0 || pos.X >= state.Board.Width || pos.Y < 0 || pos.Y >= state.Board.Height {
		return true
	}

	// Collision with any snake body (including our own)
	for _, snake := range state.Board.Snakes {
		for i, segment := range snake.Body {
			// Skip the tail unless the snake just ate (length will grow)
			if i == len(snake.Body)-1 && snake.Health != MaxHealth {
				continue
			}
			if pos.X == segment.X && pos.Y == segment.Y {
				return true
			}
		}
	}

	return false
}

// evaluateSpace uses flood fill to determine available space
func evaluateSpace(state GameState, pos Coord) float64 {
	visited := make(map[Coord]bool)
	count := floodFill(state, pos, visited, 0, state.You.Length)

	// Normalize by board size
	totalSpaces := state.Board.Width * state.Board.Height
	return float64(count) / float64(totalSpaces)
}

// floodFill recursively counts reachable spaces
func floodFill(state GameState, pos Coord, visited map[Coord]bool, depth int, maxDepth int) int {
	// Limit recursion depth for performance
	if depth > maxDepth {
		return 0
	}

	// Check if position is valid and unvisited
	if pos.X < 0 || pos.X >= state.Board.Width || pos.Y < 0 || pos.Y >= state.Board.Height {
		return 0
	}
	if visited[pos] {
		return 0
	}

	// Check if blocked by snake
	for _, snake := range state.Board.Snakes {
		for i, segment := range snake.Body {
			// Skip tails that will move
			if i == len(snake.Body)-1 {
				continue
			}
			if pos.X == segment.X && pos.Y == segment.Y {
				return 0
			}
		}
	}

	// Check if too close to enemy snake heads (they can move and cut us off)
	// Skip our own snake to avoid penalizing our own position
	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}
		// Consider positions near enemy heads as risky during space evaluation
		// Only apply this at shallow depths to avoid over-restricting distant areas
		if depth < 3 {
			dist := manhattanDistance(pos, snake.Head)
			if dist <= SpaceBufferRadius {
				// Don't completely block, but reduce the value of this space
				// by treating it as less valuable (we still count it but flag it)
				visited[pos] = true
				// Return reduced count (0.3 instead of 1.0) for risky positions
				count := 0
				count += floodFill(state, Coord{X: pos.X + 1, Y: pos.Y}, visited, depth+1, maxDepth)
				count += floodFill(state, Coord{X: pos.X - 1, Y: pos.Y}, visited, depth+1, maxDepth)
				count += floodFill(state, Coord{X: pos.X, Y: pos.Y + 1}, visited, depth+1, maxDepth)
				count += floodFill(state, Coord{X: pos.X, Y: pos.Y - 1}, visited, depth+1, maxDepth)
				// Return partial credit for this risky position but continue exploring
				return count / 3
			}
		}
	}

	visited[pos] = true
	count := 1

	// Recursively check adjacent squares
	count += floodFill(state, Coord{X: pos.X + 1, Y: pos.Y}, visited, depth+1, maxDepth)
	count += floodFill(state, Coord{X: pos.X - 1, Y: pos.Y}, visited, depth+1, maxDepth)
	count += floodFill(state, Coord{X: pos.X, Y: pos.Y + 1}, visited, depth+1, maxDepth)
	count += floodFill(state, Coord{X: pos.X, Y: pos.Y - 1}, visited, depth+1, maxDepth)

	return count
}

// evaluateFoodProximity scores based on proximity to nearest food
// Now also considers if food is dangerous due to nearby enemy snakes
func evaluateFoodProximity(state GameState, pos Coord) float64 {
	if len(state.Board.Food) == 0 {
		return 0
	}

	var targetFood Coord
	var distanceScore float64

	// Use A* for better pathfinding when health is low (more accurate around obstacles)
	if state.You.Health < HealthLow {
		food, path := findNearestFoodWithAStar(state, pos)
		if path != nil && len(path) > 0 {
			targetFood = food
			pathLength := len(path)
			if pathLength == 1 {
				distanceScore = 1.0 // Already at food
			} else {
				distanceScore = 1.0 / float64(pathLength)
			}
		} else {
			// If no path found with A*, fall back to Manhattan distance
			targetFood, distanceScore = findNearestFoodManhattan(state, pos)
		}
	} else {
		// Use Manhattan distance for non-critical situations (better performance)
		targetFood, distanceScore = findNearestFoodManhattan(state, pos)
	}

	// Check if the target food is dangerous due to nearby enemy snakes
	if isFoodDangerous(state, targetFood) {
		// Significantly reduce food attractiveness if it's near enemy snakes
		// This prevents the snake from getting trapped near enemy snakes
		distanceScore *= FoodDangerPenalty
	}

	return distanceScore
}

// findNearestFoodManhattan finds the nearest food using Manhattan distance
func findNearestFoodManhattan(state GameState, pos Coord) (Coord, float64) {
	minDist := math.MaxInt32
	var nearestFood Coord

	for _, food := range state.Board.Food {
		dist := manhattanDistance(pos, food)
		if dist < minDist {
			minDist = dist
			nearestFood = food
		}
	}

	// Inverse distance (closer is better)
	distanceScore := 1.0
	if minDist == 0 {
		distanceScore = 1.0
	} else {
		distanceScore = 1.0 / float64(minDist)
	}

	return nearestFood, distanceScore
}

// isFoodDangerous checks if food is too close to enemy snakes or in dangerous positions
// Food is considered dangerous if:
// 1. It's within FoodDangerRadius spaces of any enemy snake body segment
// 2. An enemy can reach it significantly faster than us (3+ moves)
// 3. Enemy reaches at same time and is much larger (2+ length advantage)
// 4. It's on or very near a wall when enemies exist (limits escape routes)
func isFoodDangerous(state GameState, food Coord) bool {
	myDistToFood := manhattanDistance(state.You.Head, food)
	
	// When health is critical, be more willing to take risks
	isCritical := state.You.Health < HealthCritical
	
	// Check if food is ON a wall when enemies exist
	// Food on walls is dangerous because it limits escape routes
	// Only skip this check when we're at critical health (must risk it to survive)
	// OR when we're already on/near a wall ourselves (no additional danger)
	if !isCritical && hasAnyEnemies(state) {
		// Calculate our distance from walls using helper
		ourMinDistToWall := getMinDistanceToWall(state, state.You.Head)
		
		// Check if food is directly ON any wall (distance = 0)
		foodDistFromLeft := food.X
		foodDistFromRight := state.Board.Width - 1 - food.X
		foodDistFromBottom := food.Y
		foodDistFromTop := state.Board.Height - 1 - food.Y
		foodOnWall := (foodDistFromLeft == 0 || foodDistFromRight == 0 || foodDistFromBottom == 0 || foodDistFromTop == 0)
		
		// Only mark as dangerous if food is on wall AND we're not already ON a wall
		// If we're already ON a wall (distance == 0), food on same/adjacent wall is no more dangerous
		// But if we're 1+ squares away, going to the wall is dangerous
		if foodOnWall && ourMinDistToWall >= 1 {
			return true // Food on wall and we'd have to move to/toward the wall
		}
	}
	
	for _, snake := range state.Board.Snakes {
		// Skip our own snake
		if snake.ID == state.You.ID {
			continue
		}

		// Check distance to each segment of enemy snake body
		for _, segment := range snake.Body {
			dist := manhattanDistance(food, segment)
			if dist <= FoodDangerRadius {
				return true
			}
		}
		
		// Check if enemy can reach food significantly faster than us
		// This helps avoid food baiting scenarios where enemies camp near food
		enemyDistToFood := manhattanDistance(snake.Head, food)
		
		// More aggressive: only avoid if enemy is 3+ moves closer (was 2+)
		// This allows us to contest food more often
		// Example: if we're 5 moves away and enemy is 2 moves away, that's 3 moves closer
		if enemyDistToFood < myDistToFood-2 {
			// Enemy is significantly closer - potentially dangerous
			return true
		}
		
		// If enemy can reach at same time, only avoid if they're MUCH larger (2+ length)
		// This encourages contesting food with similar-sized snakes
		// If we're critical health, be even more willing to contest
		lengthAdvantage := snake.Length - state.You.Length
		if abs(enemyDistToFood-myDistToFood) <= 1 {
			if isCritical {
				// When critical, only avoid if enemy is 3+ longer
				if lengthAdvantage >= 3 {
					return true
				}
			} else {
				// When healthy, avoid if enemy is 2+ longer
				if lengthAdvantage >= 2 {
					return true
				}
			}
		}
	}

	return false
}

// evaluateHeadCollisionRisk checks for potential head-to-head collisions
// Now enhanced with enemy move prediction to avoid dangerous positions
func evaluateHeadCollisionRisk(state GameState, pos Coord) float64 {
	risk := 0.0
	myLength := state.You.Length

	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}

		// Check if enemy snake's head could move to adjacent squares
		enemyHead := snake.Head
		for _, dir := range []string{MoveUp, MoveDown, MoveLeft, MoveRight} {
			enemyNextPos := getNextPosition(enemyHead, dir)
			
			// Skip if enemy's next position would be invalid (out of bounds or collision)
			if enemyNextPos.X < 0 || enemyNextPos.X >= state.Board.Width ||
				enemyNextPos.Y < 0 || enemyNextPos.Y >= state.Board.Height {
				continue
			}
			
			if enemyNextPos.X == pos.X && enemyNextPos.Y == pos.Y {
				// Head-to-head collision possible
				if snake.Length >= myLength {
					risk += 1.0 // Enemy is same size or larger - avoid this position
				} else {
					// Enemy is smaller - we'd win head-to-head, but still risky
					risk += 0.3
				}
			}
		}
	}

	return risk
}

// predictEnemyMoves returns all possible next positions for enemy snake heads
// This helps us avoid moving into positions where enemies can attack next turn
func predictEnemyMoves(state GameState) map[Coord][]Battlesnake {
	enemyMoves := make(map[Coord][]Battlesnake)
	
	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}
		
		// For each possible direction the enemy can move
		for _, dir := range []string{MoveUp, MoveDown, MoveLeft, MoveRight} {
			nextPos := getNextPosition(snake.Head, dir)
			
			// Skip if move would be immediately fatal for enemy
			if isImmediatelyFatal(state, nextPos) {
				continue
			}
			
			// Add this possible position
			enemyMoves[nextPos] = append(enemyMoves[nextPos], snake)
		}
	}
	
	return enemyMoves
}

// simulateMove performs a lightweight check to see if a move leaves adequate escape routes.
// This is NOT a full multi-turn simulation (too expensive), but rather a quick validation
// that the immediate next position has sufficient exit options.
// Returns true if the move appears safe, false if it may lead to a trap.
func simulateMove(state GameState, move string, turnsAhead int) bool {
	if turnsAhead <= 0 || turnsAhead > 3 {
		return true // Only look ahead 1-3 turns for performance
	}
	
	myHead := state.You.Head
	nextPos := getNextPosition(myHead, move)
	
	// Check immediate safety first
	if isImmediatelyFatal(state, nextPos) {
		return false
	}
	
	// Create a simulated future state with our snake moved
	// Note: This is a simplified simulation that doesn't handle food consumption
	// or enemy movement, focusing only on our own body position
	newBody := make([]Coord, len(state.You.Body))
	newBody[0] = nextPos // New head position
	
	// Shift body forward (snake moves, tail segment leaves)
	// NOTE: This assumes no food is eaten. If food is at nextPos, the tail
	// wouldn't move and we'd have one less escape option, making this check
	// slightly pessimistic (safer) in those cases.
	for i := 1; i < len(state.You.Body); i++ {
		newBody[i] = state.You.Body[i-1]
	}
	
	// Create simulated game state
	simState := GameState{
		Board: Board{
			Width:  state.Board.Width,
			Height: state.Board.Height,
			Snakes: make([]Battlesnake, len(state.Board.Snakes)),
			Food:   state.Board.Food, // Include food positions for more accurate checks
		},
		You: Battlesnake{
			ID:     state.You.ID,
			Body:   newBody,
			Head:   nextPos,
			Length: state.You.Length,
			Health: state.You.Health,
		},
	}
	
	// Copy snakes, updating our own
	for i, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			simState.Board.Snakes[i] = simState.You
		} else {
			simState.Board.Snakes[i] = snake
		}
	}
	
	// Check if this position leaves us with escape routes
	// Count available safe moves from new position
	safeMovesCount := 0
	for _, dir := range []string{MoveUp, MoveDown, MoveLeft, MoveRight} {
		testPos := getNextPosition(nextPos, dir)
		if !isImmediatelyFatal(simState, testPos) {
			safeMovesCount++
		}
	}
	
	// We need at least 1 escape route to survive. Requiring 2+ is conservative
	// but helps avoid getting boxed in. In desperate situations, the penalty
	// is lower than immediate death, so the snake will still take risky moves
	// if all alternatives are worse.
	return safeMovesCount >= 1
}

// evaluateCenterProximity prefers center positions
func evaluateCenterProximity(state GameState, pos Coord) float64 {
	centerX := state.Board.Width / 2
	centerY := state.Board.Height / 2
	dist := manhattanDistance(pos, Coord{X: centerX, Y: centerY})
	maxDist := centerX + centerY

	if maxDist == 0 {
		return 1.0
	}
	return 1.0 - (float64(dist) / float64(maxDist))
}

// evaluateTailProximity encourages following own tail when safe
func evaluateTailProximity(state GameState, pos Coord) float64 {
	if len(state.You.Body) < 2 {
		return 0
	}

	tail := state.You.Body[len(state.You.Body)-1]
	dist := manhattanDistance(pos, tail)

	if dist == 0 {
		return 1.0
	}
	return 1.0 / float64(dist)
}

// hasAnyEnemies checks if there are any enemy snakes on the board
// Used to determine if wall avoidance should be active
func hasAnyEnemies(state GameState) bool {
	// If there's only one snake (us), no enemies exist
	return len(state.Board.Snakes) > 1
}

// evaluateWallAvoidance returns a penalty for positions near walls/edges
// This is more aggressive than the old evaluateCornerPenalty, designed to
// prevent getting boxed in by aggressive snakes. The penalty increases
// the closer we are to walls, with a safety zone of 2-3 squares from edges.
func evaluateWallAvoidance(state GameState, pos Coord) float64 {
	// Calculate distance from each edge
	distFromLeft := pos.X
	distFromRight := state.Board.Width - 1 - pos.X
	distFromBottom := pos.Y
	distFromTop := state.Board.Height - 1 - pos.Y

	// Find minimum distance to any edge
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

	// Gradual penalty based on distance from nearest edge
	// 0 squares from edge: 1.0 penalty (maximum - on the wall)
	// 1 square from edge: 0.8 penalty (very dangerous)
	// 2 squares from edge: 0.5 penalty (still dangerous)
	// 3 squares from edge: 0.2 penalty (slight risk)
	// 4+ squares from edge: 0.0 penalty (safe)
	penalty := 0.0
	switch minDistToEdge {
	case 0:
		penalty = 1.0 // On the edge - extremely dangerous
	case 1:
		penalty = 0.8 // One square from edge - very dangerous
	case 2:
		penalty = 0.5 // Two squares from edge - moderately dangerous
	case 3:
		penalty = 0.2 // Three squares from edge - slightly dangerous
	default:
		penalty = 0.0 // Safe distance from edges
	}

	// Additional penalty for being in a corner (multiple walls close)
	// Count walls within 2 squares
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

	// If in or near a corner (2+ walls nearby), add extra penalty
	if wallsNearby >= 2 {
		penalty += 0.3 * float64(wallsNearby) // 0.6 for corner, 0.9 for very tight corner
	}

	return penalty
}

// evaluateCornerPenalty returns a penalty for positions near corners/edges
// when enemies are nearby. This prevents getting squeezed into corners.
// NOTE: This function is kept for backward compatibility but is now superseded
// by evaluateWallAvoidance which is more aggressive.
func evaluateCornerPenalty(state GameState, pos Coord) float64 {
	// Calculate distance from edges
	distFromLeft := pos.X
	distFromRight := state.Board.Width - 1 - pos.X
	distFromBottom := pos.Y
	distFromTop := state.Board.Height - 1 - pos.Y

	// Find minimum distance to any edge
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

	// Count how many directions are blocked (by walls or approaching edge)
	blockedDirections := 0

	// Check if near walls (within 1-2 squares)
	if distFromLeft <= 1 {
		blockedDirections++
	}
	if distFromRight <= 1 {
		blockedDirections++
	}
	if distFromBottom <= 1 {
		blockedDirections++
	}
	if distFromTop <= 1 {
		blockedDirections++
	}

	// Calculate penalty based on how cornered we are
	// 0 blocked directions = no penalty
	// 1 blocked direction = slight penalty (edge)
	// 2+ blocked directions = high penalty (corner or very limited escape)
	penalty := 0.0
	switch blockedDirections {
	case 0:
		penalty = 0.0
	case 1:
		penalty = 0.3 // Slight penalty for being near an edge
	case 2:
		penalty = 0.8 // High penalty for corner
	default:
		penalty = 1.0 // Maximum penalty for severely cornered position
	}

	// Also factor in overall distance from edges - being far from edges is safer
	if minDistToEdge <= 1 {
		penalty += 0.2
	}

	return penalty
}

// manhattanDistance calculates the Manhattan distance between two coordinates
func manhattanDistance(a, b Coord) int {
	return abs(a.X-b.X) + abs(a.Y-b.Y)
}

// abs returns the absolute value of an integer
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// hasEnemiesNearby checks if any enemy snakes are within EnemyProximityRadius
// of our head. Returns true if enemies are nearby, indicating we should avoid
// circling/tail-chasing behavior to prevent becoming a sitting target.
func hasEnemiesNearby(state GameState) bool {
	myHead := state.You.Head

	for _, snake := range state.Board.Snakes {
		// Skip our own snake
		if snake.ID == state.You.ID {
			continue
		}

		// Check if enemy head is within proximity radius
		dist := manhattanDistance(myHead, snake.Head)
		if dist <= EnemyProximityRadius {
			return true
		}
	}

	return false
}

// isBeingChased detects if an enemy snake is following us closely
// Returns true if we should take evasive action
func isBeingChased(state GameState) bool {
	myHead := state.You.Head
	myTail := state.You.Body[len(state.You.Body)-1]
	
	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}
		
		// Check if enemy head is close to our tail
		distToTail := manhattanDistance(snake.Head, myTail)
		distToHead := manhattanDistance(snake.Head, myHead)
		
		// Enemy is chasing if they're closer to our tail than our head
		// and within a reasonable distance
		if distToTail <= 3 && distToTail < distToHead {
			return true
		}
	}
	
	return false
}

// detectCutoff checks if enemy snakes are blocking our escape routes
// Returns a penalty score based on how trapped we are
func detectCutoff(state GameState, pos Coord) float64 {
	// Count how many valid moves we have from this position
	validMoves := 0
	blockedByEnemies := 0
	
	for _, dir := range []string{MoveUp, MoveDown, MoveLeft, MoveRight} {
		nextPos := getNextPosition(pos, dir)
		
		// Check if move is valid
		if nextPos.X < 0 || nextPos.X >= state.Board.Width ||
			nextPos.Y < 0 || nextPos.Y >= state.Board.Height {
			continue
		}
		
		// Check if blocked by snake
		blocked := false
		blockedByEnemy := false
		for _, snake := range state.Board.Snakes {
			for i, segment := range snake.Body {
				// Skip tails that will move
				if i == len(snake.Body)-1 && snake.Health != MaxHealth {
					continue
				}
				if nextPos.X == segment.X && nextPos.Y == segment.Y {
					blocked = true
					if snake.ID != state.You.ID {
						blockedByEnemy = true
					}
					break
				}
			}
			if blocked {
				break
			}
		}
		
		if !blocked {
			validMoves++
		} else if blockedByEnemy {
			blockedByEnemies++
		}
	}
	
	// Calculate cutoff penalty
	// If we have 0-1 valid moves, we're in serious danger
	if validMoves == 0 {
		return 10.0 // Extreme danger - completely trapped
	} else if validMoves == 1 {
		return 5.0 // High danger - only one escape route
	} else if validMoves == 2 && blockedByEnemies >= 2 {
		return 2.0 // Moderate danger - limited options
	}
	
	return 0.0
}

// calculateAggressionScore determines how aggressive we should be based on current game state
// Returns a score from 0.0 (defensive) to 1.0 (aggressive)
func calculateAggressionScore(state GameState) float64 {
	score := 0.5 // Start neutral
	
	myLength := state.You.Length
	myHealth := state.You.Health
	
	// Health factor: increase aggression when healthy
	if myHealth >= AggressionHealthThreshold {
		score += 0.2
	} else if myHealth < HealthCritical {
		score -= 0.3 // Very defensive when low health
	}
	
	// Length advantage factor
	if len(state.Board.Snakes) > 1 {
		totalEnemyLength := 0
		enemyCount := 0
		longestEnemy := 0
		
		for _, snake := range state.Board.Snakes {
			if snake.ID != state.You.ID {
				totalEnemyLength += snake.Length
				enemyCount++
				if snake.Length > longestEnemy {
					longestEnemy = snake.Length
				}
			}
		}
		
		if enemyCount > 0 {
			avgEnemyLength := float64(totalEnemyLength) / float64(enemyCount)
			
			// Compare to longest enemy (most dangerous)
			if myLength > longestEnemy+AggressionLengthAdvantage {
				score += 0.3 // Much longer than strongest enemy
			} else if myLength > longestEnemy {
				score += 0.1 // Slightly longer
			} else if myLength < longestEnemy-AggressionLengthAdvantage {
				score -= 0.2 // Much shorter, be defensive
			}
			
			// Compare to average (overall dominance)
			if float64(myLength) > avgEnemyLength+1 {
				score += 0.1
			}
		}
	}
	
	// Space control factor: increase aggression when we have more space
	mySpace := evaluateSpace(state, state.You.Head)
	if mySpace > 0.4 {
		score += 0.1 // We control a lot of space
	} else if mySpace < 0.2 {
		score -= 0.2 // Limited space, be defensive
	}
	
	// Board position factor: reduce aggression near walls
	distToWall := getMinDistanceToWall(state, state.You.Head)
	if distToWall <= 1 {
		score -= 0.1
	}
	
	// Clamp between 0 and 1
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}
	
	return score
}

// getMinDistanceToWall returns the minimum distance to any wall
func getMinDistanceToWall(state GameState, pos Coord) int {
	distFromLeft := pos.X
	distFromRight := state.Board.Width - 1 - pos.X
	distFromBottom := pos.Y
	distFromTop := state.Board.Height - 1 - pos.Y
	
	minDist := distFromLeft
	if distFromRight < minDist {
		minDist = distFromRight
	}
	if distFromBottom < minDist {
		minDist = distFromBottom
	}
	if distFromTop < minDist {
		minDist = distFromTop
	}
	
	return minDist
}

// minInt returns the minimum of multiple integers
func minInt(values ...int) int {
	if len(values) == 0 {
		return 0
	}
	min := values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

// evaluateEnemyReachableSpace calculates the reachable space for an enemy snake from a given position
// This is used to determine if we're successfully trapping them
// Note: Creates a new visited map for each call. This is called up to 8 times per turn
// (4 moves × 2 enemy space calculations). Total allocation: ~1KB per turn, acceptable overhead.
func evaluateEnemyReachableSpace(state GameState, enemy Battlesnake, fromPos Coord) int {
	visited := make(map[Coord]bool)
	return floodFillForSnake(state, fromPos, visited, 0, enemy.Length, enemy.ID)
}

// floodFillForSnake is similar to floodFill but considers a specific snake's perspective
// It's used to calculate how much space an enemy snake can reach
// Note: Uses recursion with depth limit (snake length, typically 3-20). This is safe and
// won't cause stack overflow. An iterative implementation with a queue would be more complex
// without significant performance benefit given the small depth limits.
func floodFillForSnake(state GameState, pos Coord, visited map[Coord]bool, depth int, maxDepth int, snakeID string) int {
	// Limit recursion depth for performance and stack safety
	if depth > maxDepth {
		return 0
	}

	// Check if position is valid and unvisited
	if pos.X < 0 || pos.X >= state.Board.Width || pos.Y < 0 || pos.Y >= state.Board.Height {
		return 0
	}
	if visited[pos] {
		return 0
	}

	// Check if blocked by snake (excluding the snake's own tail which will move)
	for _, snake := range state.Board.Snakes {
		for i, segment := range snake.Body {
			// Skip the tail of the snake we're evaluating for (it will move)
			if snake.ID == snakeID && i == len(snake.Body)-1 {
				continue
			}
			// Skip other snakes' tails
			if i == len(snake.Body)-1 {
				continue
			}
			if pos.X == segment.X && pos.Y == segment.Y {
				return 0
			}
		}
	}

	visited[pos] = true
	count := 1

	// Recursively check adjacent squares
	count += floodFillForSnake(state, Coord{X: pos.X + 1, Y: pos.Y}, visited, depth+1, maxDepth, snakeID)
	count += floodFillForSnake(state, Coord{X: pos.X - 1, Y: pos.Y}, visited, depth+1, maxDepth, snakeID)
	count += floodFillForSnake(state, Coord{X: pos.X, Y: pos.Y + 1}, visited, depth+1, maxDepth, snakeID)
	count += floodFillForSnake(state, Coord{X: pos.X, Y: pos.Y - 1}, visited, depth+1, maxDepth, snakeID)

	return count
}

// evaluateTrapOpportunity checks if a move would trap an enemy snake
// Returns a trap score (higher is better) or 0 if not a trap opportunity
func evaluateTrapOpportunity(state GameState, nextPos Coord) float64 {
	trapScore := 0.0
	
	// For each enemy, check if this move reduces their available space significantly
	for _, enemy := range state.Board.Snakes {
		if enemy.ID == state.You.ID {
			continue
		}
		
		// Calculate enemy's current reachable space
		enemyCurrentSpace := evaluateEnemyReachableSpace(state, enemy, enemy.Head)
		
		// Simulate the game state after our move
		simState := simulateGameState(state, state.You.ID, nextPos)
		
		// Calculate enemy's reachable space after our move
		enemyNewSpace := evaluateEnemyReachableSpace(simState, enemy, enemy.Head)
		
		// Check if we're significantly reducing their space
		spaceReduction := float64(enemyCurrentSpace - enemyNewSpace)
		totalSpaces := float64(state.Board.Width * state.Board.Height)
		spaceReductionPercent := spaceReduction / totalSpaces
		
		// Only consider it a trap if:
		// 1. We reduce their space significantly (> TrapSpaceThreshold)
		// 2. We maintain enough space for ourselves (TrapSafetyMargin)
		// 3. The enemy is not significantly larger (would be dangerous)
		if spaceReductionPercent > TrapSpaceThreshold {
			// Safety check: ensure we have enough space
			mySimSpace := evaluateSpace(simState, nextPos)
			enemySimSpace := float64(enemyNewSpace) / totalSpaces
			
			if mySimSpace > enemySimSpace*TrapSafetyMargin {
				// Safe trap opportunity
				// Score higher if enemy is smaller or same size
				if enemy.Length <= state.You.Length {
					trapScore += spaceReductionPercent * 2.0
				} else if enemy.Length <= state.You.Length+2 {
					trapScore += spaceReductionPercent * 1.0
				}
			}
		}
	}
	
	return trapScore
}

// simulateGameState creates a new game state with a snake moved to a new position
// This is used for trap detection to see what the board would look like after our move
func simulateGameState(state GameState, snakeID string, newHead Coord) GameState {
	simState := GameState{
		Game:  state.Game,
		Turn:  state.Turn + 1,
		Board: Board{
			Width:   state.Board.Width,
			Height:  state.Board.Height,
			Food:    state.Board.Food,
			Hazards: state.Board.Hazards,
			Snakes:  make([]Battlesnake, len(state.Board.Snakes)),
		},
		You: state.You,
	}
	
	// Copy snakes and update the specified snake
	for i, snake := range state.Board.Snakes {
		if snake.ID == snakeID {
			// Create new body with new head and shifted body
			newBody := make([]Coord, len(snake.Body))
			newBody[0] = newHead
			for j := 1; j < len(snake.Body); j++ {
				newBody[j] = snake.Body[j-1]
			}
			
			simState.Board.Snakes[i] = Battlesnake{
				ID:     snake.ID,
				Name:   snake.Name,
				Health: snake.Health,
				Body:   newBody,
				Head:   newHead,
				Length: snake.Length,
			}
			
			// Update You if this is our snake
			if snake.ID == state.You.ID {
				simState.You = simState.Board.Snakes[i]
			}
		} else {
			simState.Board.Snakes[i] = snake
		}
	}
	
	return simState
}

// getNextPosition returns the coordinate resulting from a move
func getNextPosition(pos Coord, move string) Coord {
	switch move {
	case MoveUp:
		return Coord{X: pos.X, Y: pos.Y + 1}
	case MoveDown:
		return Coord{X: pos.X, Y: pos.Y - 1}
	case MoveLeft:
		return Coord{X: pos.X - 1, Y: pos.Y}
	case MoveRight:
		return Coord{X: pos.X + 1, Y: pos.Y}
	}
	return pos
}

// A* Pathfinding Implementation

// aStarNode represents a node in the A* search
type aStarNode struct {
	pos    Coord
	gScore int // Cost from start
	fScore int // gScore + heuristic to goal
	parent *aStarNode
	index  int // Index in the priority queue
}

// priorityQueue implements heap.Interface for A* algorithm
type priorityQueue []*aStarNode

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	return pq[i].fScore < pq[j].fScore
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueue) Push(x interface{}) {
	n := len(*pq)
	node := x.(*aStarNode)
	node.index = n
	*pq = append(*pq, node)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	node := old[n-1]
	old[n-1] = nil
	node.index = -1
	*pq = old[0 : n-1]
	return node
}

// aStarSearch finds the shortest path from start to goal using A* algorithm
// Returns nil if no path exists or search exceeds maxNodes
func aStarSearch(state GameState, start, goal Coord, maxNodes int) []Coord {
	// Check if goal is blocked
	if isPositionBlocked(state, goal) {
		return nil
	}

	// Check if start equals goal
	if start.X == goal.X && start.Y == goal.Y {
		return []Coord{start}
	}

	// Initialize open set with start node
	openSet := &priorityQueue{}
	heap.Init(openSet)

	startNode := &aStarNode{
		pos:    start,
		gScore: 0,
		fScore: manhattanDistance(start, goal),
		parent: nil,
	}
	heap.Push(openSet, startNode)

	// Track visited nodes and their best scores
	visited := make(map[Coord]int)
	nodesExplored := 0

	for openSet.Len() > 0 && nodesExplored < maxNodes {
		// Get node with lowest f-score
		current := heap.Pop(openSet).(*aStarNode)
		nodesExplored++

		// Check if we reached the goal
		if current.pos.X == goal.X && current.pos.Y == goal.Y {
			return reconstructPath(current)
		}

		// Mark as visited
		visited[current.pos] = current.gScore

		// Explore neighbors
		for _, neighbor := range getValidNeighbors(state, current.pos) {
			tentativeGScore := current.gScore + 1

			// Skip if we've found a better path to this neighbor
			if prevScore, seen := visited[neighbor]; seen && prevScore <= tentativeGScore {
				continue
			}

			// Add neighbor to open set
			neighborNode := &aStarNode{
				pos:    neighbor,
				gScore: tentativeGScore,
				fScore: tentativeGScore + manhattanDistance(neighbor, goal),
				parent: current,
			}
			heap.Push(openSet, neighborNode)
		}
	}

	// No path found
	return nil
}

// reconstructPath builds the path from start to goal by following parent pointers
func reconstructPath(node *aStarNode) []Coord {
	path := []Coord{}
	current := node

	// Build path in reverse (goal to start)
	for current != nil {
		path = append(path, current.pos)
		current = current.parent
	}

	// Reverse the path to get start to goal order
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}

	return path
}

// getValidNeighbors returns adjacent positions that are not blocked
func getValidNeighbors(state GameState, pos Coord) []Coord {
	neighbors := []Coord{}
	directions := []Coord{
		{X: 0, Y: 1},  // up
		{X: 0, Y: -1}, // down
		{X: -1, Y: 0}, // left
		{X: 1, Y: 0},  // right
	}

	for _, dir := range directions {
		newPos := Coord{X: pos.X + dir.X, Y: pos.Y + dir.Y}

		// Check bounds
		if newPos.X < 0 || newPos.X >= state.Board.Width ||
			newPos.Y < 0 || newPos.Y >= state.Board.Height {
			continue
		}

		// Check if blocked by snake
		if isPositionBlocked(state, newPos) {
			continue
		}

		neighbors = append(neighbors, newPos)
	}

	return neighbors
}

// isPositionBlocked checks if a position is occupied by a snake body
func isPositionBlocked(state GameState, pos Coord) bool {
	for _, snake := range state.Board.Snakes {
		for i, segment := range snake.Body {
			// Skip tails that will move (snake hasn't just eaten)
			// When a snake eats food, its health is reset to MaxHealth and
			// the tail doesn't move on that turn (body grows). In all other
			// cases, the tail moves forward so that position will be empty.
			if i == len(snake.Body)-1 && snake.Health != MaxHealth {
				continue
			}

			if pos.X == segment.X && pos.Y == segment.Y {
				return true
			}
		}
	}
	return false
}

// findNearestFoodWithAStar finds the closest reachable food using A*
func findNearestFoodWithAStar(state GameState, pos Coord) (Coord, []Coord) {
	var nearestFood Coord
	var shortestPath []Coord
	shortestLength := math.MaxInt32

	// Try A* to each food item, keeping track of shortest path
	for _, food := range state.Board.Food {
		path := aStarSearch(state, pos, food, MaxAStarNodes)
		if path != nil && len(path) < shortestLength {
			shortestPath = path
			nearestFood = food
			shortestLength = len(path)
		}
	}

	return nearestFood, shortestPath
}

// evaluateSpaceRatio calculates the ratio of available space to snake body length
// This is used for ratio-based trap detection as implemented in the baseline snake
// Returns the ratio (e.g., 0.5 = 50% of body length in available space)
func evaluateSpaceRatio(state GameState, pos Coord) float64 {
	visited := make(map[Coord]bool)
	spaceCount := floodFill(state, pos, visited, 0, state.You.Length*2) // Increased maxDepth for better accuracy
	bodyLength := float64(state.You.Length)
	
	if bodyLength == 0 {
		return 1.0
	}
	
	ratio := float64(spaceCount) / bodyLength
	return ratio
}

// evaluateTrapPenalty returns a penalty based on space-to-body-length ratio
// Implements graduated trap detection similar to baseline snake:
// - Critical trap (< 40%): -400 (reduced from -600 to not be too conservative)
// - Severe trap (40-60%): -300 (reduced from -450)
// - Moderate trap (60-80%): -150 (reduced from -250)
// - Good space (80%+): 0
func evaluateTrapPenalty(state GameState, pos Coord) float64 {
	ratio := evaluateSpaceRatio(state, pos)
	
	if ratio < 0.40 {
		// Critical trap - very dangerous
		return 400.0
	} else if ratio < 0.60 {
		// Severe trap - bad situation
		return 300.0
	} else if ratio < 0.80 {
		// Moderate trap - concerning
		return 150.0
	}
	
	// Good space - no penalty
	return 0.0
}

// evaluateFoodDeathTrap checks if eating food would trap the snake
// Simulates eating the food (tail doesn't move) and checks remaining space
// Returns true if it's a food death trap (< 70% of body length in remaining space)
func evaluateFoodDeathTrap(state GameState, pos Coord, foodPos Coord) float64 {
	// Only check if this position is food
	isFood := false
	for _, food := range state.Board.Food {
		if pos.X == food.X && pos.Y == food.Y {
			isFood = true
			break
		}
	}
	
	if !isFood {
		return 0.0 // Not eating food, no trap risk
	}
	
	// Simulate the state after eating food
	// When eating food, the snake grows (tail doesn't move)
	simState := GameState{
		Game:  state.Game,
		Turn:  state.Turn,
		Board: state.Board,
		You:   state.You,
	}
	
	// Update our snake to have the new head (simulating the move)
	// but keep the tail (simulating growth from eating)
	newBody := make([]Coord, len(state.You.Body)+1)
	newBody[0] = pos
	copy(newBody[1:], state.You.Body)
	
	simState.You = Battlesnake{
		ID:     state.You.ID,
		Name:   state.You.Name,
		Health: MaxHealth,
		Body:   newBody,
		Head:   pos,
		Length: state.You.Length + 1,
	}
	
	// Update the board snakes as well
	for i, snake := range simState.Board.Snakes {
		if snake.ID == state.You.ID {
			simState.Board.Snakes[i] = simState.You
			break
		}
	}
	
	// Calculate available space after eating
	visited := make(map[Coord]bool)
	spaceCount := floodFill(simState, pos, visited, 0, simState.You.Length*2)
	ratio := float64(spaceCount) / float64(simState.You.Length)
	
	// Baseline uses 70% threshold for food death traps
	// Using 500 instead of 800 to not be overly conservative
	if ratio < 0.70 {
		// This food would trap us - apply strong penalty
		return 500.0
	}
	
	return 0.0
}

// evaluateOneMoveLooka head checks if a move leads to a dead end
// Simulates one move ahead and checks if the worst-case future space ratio drops significantly
// Returns penalty if the move leads to reduced future options
func evaluateOneMoveAhead(state GameState, move string) float64 {
	myHead := state.You.Head
	nextPos := getNextPosition(myHead, move)
	
	// Get current space ratio
	currentRatio := evaluateSpaceRatio(state, myHead)
	
	// Simulate all 4 possible moves from next position
	possibleMoves := []string{MoveUp, MoveDown, MoveLeft, MoveRight}
	worstFutureRatio := 1.0
	
	for _, futureMove := range possibleMoves {
		futurePos := getNextPosition(nextPos, futureMove)
		
		// Skip if immediately fatal
		if isImmediatelyFatal(state, futurePos) {
			continue
		}
		
		// Calculate space ratio for this future position
		futureRatio := evaluateSpaceRatio(state, futurePos)
		
		if futureRatio < worstFutureRatio {
			worstFutureRatio = futureRatio
		}
	}
	
	// If worst future ratio is less than 80% of current ratio, apply penalty
	// This indicates we're moving into a position with limited future options
	// Using 100 instead of 200 to not be overly conservative
	if worstFutureRatio < currentRatio * 0.80 {
		return 100.0
	}
	
	return 0.0
}
