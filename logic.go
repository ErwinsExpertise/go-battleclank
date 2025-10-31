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

	// NEW: Check for self-trapping scenarios (multi-turn lookahead)
	// This helps prevent the snake from moving into positions where it will
	// collide with itself in the next few turns.
	// Note: This is a lightweight check that only validates escape routes exist,
	// not a full recursive simulation, to maintain performance.
	if !simulateMove(state, move, 1) {
		// This move leads to a self-trap situation - apply strong penalty
		// but not as severe as immediate death to allow it as a last resort
		score -= 3000.0
	}

	// NEW: Check if we're moving into enemy danger zones
	// Predict where enemies can move next turn
	enemyMoves := predictEnemyMoves(state)
	if enemies, inDanger := enemyMoves[nextPos]; inDanger {
		// We're moving into a position an enemy could also move to
		for _, enemy := range enemies {
			if enemy.Length >= state.You.Length {
				// Enemy is same size or larger - very dangerous
				score -= 800.0
			} else {
				// Enemy is smaller - less dangerous but still risky
				score -= 200.0
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
	// Reduced when we're significantly smaller than nearby enemies (defensive play)
	foodFactor := evaluateFoodProximity(state, nextPos)
	foodWeight := 0.0
	
	// Check if we're outmatched by nearby enemies
	outmatched := isOutmatchedByNearbyEnemies(state)
	
	if state.You.Health < HealthCritical {
		// Critical health: aggressive food seeking (unless suicidal)
		foodWeight = 300.0
		if outmatched {
			foodWeight = 200.0 // Still seek food but more cautiously
		}
	} else if state.You.Health < HealthLow {
		// Low health: strong food seeking
		foodWeight = 200.0
		if outmatched {
			foodWeight = 100.0
		}
	} else {
		// Healthy: moderate food seeking to prevent circling
		foodWeight = 50.0
		if outmatched {
			foodWeight = 30.0 // Reduce food seeking when outmatched
		}
	}
	score += foodFactor * foodWeight

	// Avoid enemy snakes' heads (they might kill us in head-to-head)
	headCollisionRisk := evaluateHeadCollisionRisk(state, nextPos)
	score -= headCollisionRisk * 500.0

	// Prefer center positions early game
	if state.Turn < 50 {
		centerFactor := evaluateCenterProximity(state, nextPos)
		score += centerFactor * 10.0
	}

	// Tail chasing when safe - but not when enemies are nearby
	// When enemies are nearby, avoid circling to prevent becoming a sitting target
	if state.You.Health > HealthCritical && !hasEnemiesNearby(state) {
		tailFactor := evaluateTailProximity(state, nextPos)
		score += tailFactor * 50.0
	}

	// Penalize corner/edge positions when enemies exist (not just nearby)
	// This prevents getting squeezed into corners with limited escape routes
	// For extremely aggressive snakes, we need to avoid walls at all costs
	if hasAnyEnemies(state) {
		wallPenalty := evaluateWallAvoidance(state, nextPos)
		score -= wallPenalty * 400.0
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
			if snake.Length > myLength+2 {
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

// isFoodDangerous checks if food is too close to enemy snakes
// Food is considered dangerous if:
// 1. It's within FoodDangerRadius spaces of any enemy snake body segment
// 2. An enemy can reach it faster than us
// 3. We don't have escape routes after getting the food
func isFoodDangerous(state GameState, food Coord) bool {
	myDistToFood := manhattanDistance(state.You.Head, food)
	
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
		
		// Only consider it dangerous if enemy is noticeably closer (2+ moves)
		// This avoids false positives where enemy happens to be slightly closer
		// but isn't actually camping or baiting
		if enemyDistToFood < myDistToFood-1 {
			// Enemy is significantly closer - potentially dangerous
			return true
		}
		
		// If enemy can reach at same time/±1 move and is larger, be cautious
		if abs(enemyDistToFood-myDistToFood) <= 1 && snake.Length >= state.You.Length {
			return true
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
