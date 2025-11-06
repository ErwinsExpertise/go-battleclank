package heuristics

import (
	"github.com/ErwinsExpertise/go-battleclank/config"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"math"
)

// Advanced tactical behaviors for continuous training exploration

const (
	// InwardTrapMinEnemyLength is the minimum enemy length to attempt inward trapping
	InwardTrapMinEnemyLength = 5

	// InwardTrapCenterRadius defines the center region for trapping
	InwardTrapCenterRadius = 2

	// AggressiveSpaceControlThreshold is the early game turn threshold
	AggressiveSpaceControlThreshold = 50

	// EdgeParallelDistance is the distance from boundary to trigger parallel avoidance
	EdgeParallelDistance = 2
)

// EvaluateInwardTrap attempts to trap enemy in center by surrounding from outside
// Only activates when: 1) We're longer than enemy, AND 2) We're closer to center
// Returns bonus score if this move helps create an inward trap
func EvaluateInwardTrap(state *board.GameState, nextPos board.Coord) float64 {
	score := 0.0
	centerX := state.Board.Width / 2
	centerY := state.Board.Height / 2
	myHeadDistToCenter := manhattanDistance(state.You.Head, board.Coord{X: centerX, Y: centerY})

	for _, enemy := range state.Board.Snakes {
		if enemy.ID == state.You.ID {
			continue
		}

		// Only trap longer snakes (self-collision is a real risk)
		if enemy.Length < InwardTrapMinEnemyLength {
			continue
		}

		// Check if enemy is in center region
		enemyDistToCenter := manhattanDistance(enemy.Head, board.Coord{X: centerX, Y: centerY})
		if enemyDistToCenter > InwardTrapCenterRadius {
			continue // Enemy not in center, no inward trap opportunity
		}

		// NEW: Only activate when we're LONGER than enemy
		if state.You.Length <= enemy.Length {
			continue // Don't try to trap snakes that are same size or larger
		}

		// NEW: Only activate when we're CLOSER to center than enemy
		if myHeadDistToCenter >= enemyDistToCenter {
			continue // We're not closer to center, can't effectively trap
		}

		// Check if our move positions us on the perimeter (surrounding)
		ourDistToCenter := manhattanDistance(nextPos, board.Coord{X: centerX, Y: centerY})

		// We want to be on the outside (further from center than enemy)
		if ourDistToCenter > enemyDistToCenter {
			// Bonus for positioning to surround
			surroundBonus := float64(enemyDistToCenter) / float64(ourDistToCenter) * 100.0

			// Extra bonus if we're cutting off escape routes
			escapeRoutes := countEnemyEscapeRoutes(state, enemy, centerX, centerY)
			if escapeRoutes <= 2 {
				surroundBonus *= 1.5
			}

			score += surroundBonus
		}
	}

	return score
}

// EvaluateAggressiveSpaceControl prioritizes territory denial in early game
// Returns bonus for moves that claim valuable board regions
func EvaluateAggressiveSpaceControl(state *board.GameState, nextPos board.Coord) float64 {
	// Only aggressive in early game
	if state.Turn > AggressiveSpaceControlThreshold {
		return 0.0
	}

	score := 0.0

	// Value positions near center higher (more strategic)
	centerX := state.Board.Width / 2
	centerY := state.Board.Height / 2
	distToCenter := manhattanDistance(nextPos, board.Coord{X: centerX, Y: centerY})
	maxDist := float64(state.Board.Width + state.Board.Height)

	// Closer to center = higher score in early game
	centerScore := (1.0 - float64(distToCenter)/maxDist) * 50.0
	score += centerScore

	// Bonus for positions that block enemy access to center
	for _, enemy := range state.Board.Snakes {
		if enemy.ID == state.You.ID {
			continue
		}

		enemyDistToCenter := manhattanDistance(enemy.Head, board.Coord{X: centerX, Y: centerY})
		ourDistToCenter := manhattanDistance(nextPos, board.Coord{X: centerX, Y: centerY})

		// If we're between enemy and center, bonus
		if ourDistToCenter < enemyDistToCenter {
			distToEnemy := manhattanDistance(nextPos, enemy.Head)
			if distToEnemy < 4 {
				score += 30.0 // Blocking bonus
			}
		}
	}

	return score
}

// EvaluatePredictiveHeadOnAvoidance uses velocity vector prediction
// Predicts where enemy heads will be and avoids those positions
func EvaluatePredictiveHeadOnAvoidance(state *board.GameState, nextPos board.Coord) float64 {
	penalty := 0.0

	for _, enemy := range state.Board.Snakes {
		if enemy.ID == state.You.ID || len(enemy.Body) < 2 {
			continue
		}

		// Calculate enemy velocity (direction of last move)
		enemyVelocity := board.Coord{
			X: enemy.Body[0].X - enemy.Body[1].X,
			Y: enemy.Body[0].Y - enemy.Body[1].Y,
		}

		// Predict enemy's next position based on velocity
		predictedPos := board.Coord{
			X: enemy.Head.X + enemyVelocity.X,
			Y: enemy.Head.Y + enemyVelocity.Y,
		}

		// Check if our move would collide with predicted enemy position
		if nextPos.X == predictedPos.X && nextPos.Y == predictedPos.Y {
			// Potential head-on collision
			if enemy.Length >= state.You.Length {
				// We would lose or draw
				penalty += 500.0
			} else {
				// We would win, but still risky
				penalty += 100.0
			}
		}

		// Also check adjacent positions (enemy might turn)
		adjacentToPredicted := manhattanDistance(nextPos, predictedPos)
		if adjacentToPredicted == 1 {
			if enemy.Length >= state.You.Length {
				penalty += 200.0 // High risk zone
			}
		}
	}

	return penalty
}

// EvaluateEnergyConservation encourages midgame efficiency
// Reduces unnecessary movement when health is stable
func EvaluateEnergyConservation(state *board.GameState, nextPos board.Coord) float64 {
	// Only apply in midgame (turns 50-150)
	if state.Turn < 50 || state.Turn > 150 {
		return 0.0
	}

	// Only conserve if health is comfortable
	if state.You.Health < 50 {
		return 0.0
	}

	bonus := 0.0

	// Prefer moves that stay in safe, spacious areas
	spaceAtPos := FloodFill(state, nextPos, 50)
	if spaceAtPos > state.You.Length*3 {
		bonus += 20.0 // Plenty of space, can afford to be patient
	}

	// Small penalty for unnecessary food seeking when health is good
	if state.You.Health > 70 {
		nearestFood := findNearestFood(state, nextPos)
		if nearestFood != nil {
			distToFood := manhattanDistance(nextPos, *nearestFood)
			// If moving toward food but health is high, small penalty
			prevDist := manhattanDistance(state.You.Head, *nearestFood)
			if distToFood < prevDist {
				bonus -= 5.0 // Minor penalty for food seeking at high health
			}
		}
	}

	return bonus
}

// EvaluateAdaptiveWallHugging balances safe wall hugging vs constriction-based
// Uses walls strategically rather than avoiding them completely
func EvaluateAdaptiveWallHugging(state *board.GameState, nextPos board.Coord) float64 {
	score := 0.0

	// Check if position is near wall
	nearWall := nextPos.X == 0 || nextPos.X == state.Board.Width-1 ||
		nextPos.Y == 0 || nextPos.Y == state.Board.Height-1

	if !nearWall {
		return 0.0
	}

	// Wall hugging is good when:
	// 1. We're smaller than enemies (safety)
	// 2. Low health and need to conserve space
	// 3. Late game positioning

	avgEnemyLength := 0
	enemyCount := 0
	for _, enemy := range state.Board.Snakes {
		if enemy.ID != state.You.ID {
			avgEnemyLength += enemy.Length
			enemyCount++
		}
	}

	if enemyCount > 0 {
		avgEnemyLength /= enemyCount

		// If we're smaller, walls provide safety
		if state.You.Length < avgEnemyLength {
			score += 30.0
		}
	}

	// In late game (1v1 or 2 snakes), walls can be strategic
	if len(state.Board.Snakes) <= 2 {
		score += 20.0
	}

	// But avoid if it significantly reduces our space
	spaceAtPos := FloodFill(state, nextPos, 30)
	if spaceAtPos < state.You.Length*2 {
		score -= 50.0 // Too constricted
	}

	return score
}

// Helper functions

func manhattanDistance(a, b board.Coord) int {
	return int(math.Abs(float64(a.X-b.X)) + math.Abs(float64(a.Y-b.Y)))
}

func countEnemyEscapeRoutes(state *board.GameState, enemy board.Snake, centerX, centerY int) int {
	routes := 0
	directions := []board.Coord{
		{X: 0, Y: 1},  // up
		{X: 0, Y: -1}, // down
		{X: -1, Y: 0}, // left
		{X: 1, Y: 0},  // right
	}

	for _, dir := range directions {
		nextPos := board.Coord{X: enemy.Head.X + dir.X, Y: enemy.Head.Y + dir.Y}

		// Check if this direction leads away from center (escape route)
		currentDist := manhattanDistance(enemy.Head, board.Coord{X: centerX, Y: centerY})
		nextDist := manhattanDistance(nextPos, board.Coord{X: centerX, Y: centerY})

		if nextDist > currentDist {
			// Leads away from center
			// Check if position is valid and not occupied (skip tails since they'll move)
			skipTails := true
			if state.Board.IsInBounds(nextPos) && !state.Board.IsOccupied(nextPos, skipTails) {
				routes++
			}
		}
	}

	return routes
}

func findNearestFood(state *board.GameState, pos board.Coord) *board.Coord {
	if len(state.Board.Food) == 0 {
		return nil
	}

	nearest := &state.Board.Food[0]
	minDist := manhattanDistance(pos, *nearest)

	for i := 1; i < len(state.Board.Food); i++ {
		dist := manhattanDistance(pos, state.Board.Food[i])
		if dist < minDist {
			minDist = dist
			nearest = &state.Board.Food[i]
		}
	}

	return nearest
}

// DetectCutoffKill checks if we can execute a guaranteed cutoff kill
// This is a deterministic, high-priority strategy that bypasses all scoring
//
// Trigger conditions (all must be true):
// 1. Both snakes moving in same direction
// 2. Opponent on wall side, we're one lane inward
// 3. Our head at least 2 tiles ahead of opponent's head
// 4. Our body extends at least 2 tiles past opponent's head position
//
// Returns the cutoff move direction if conditions are met, empty string otherwise
func DetectCutoffKill(state *board.GameState) string {
	myHead := state.You.Head

	// Need at least 2 body segments to determine direction
	if len(state.You.Body) < 2 {
		return ""
	}

	myNeck := state.You.Body[1]
	myDirection := getDirection(myHead, myNeck)

	if myDirection == "" {
		return ""
	}

	// Check each enemy snake
	for _, enemy := range state.Board.Snakes {
		if enemy.ID == state.You.ID {
			continue
		}

		// Enemy needs at least 2 segments to determine direction
		if len(enemy.Body) < 2 {
			continue
		}

		enemyNeck := enemy.Body[1]
		enemyDirection := getDirection(enemy.Head, enemyNeck)

		// Condition 1: Both moving in same direction
		if myDirection != enemyDirection {
			continue
		}

		// Determine if we're moving parallel to a wall
		// Check if enemy is on the wall and we're one lane inward
		var cutoffMove string
		var enemyIsOnWall bool
		var weAreInward bool

		switch myDirection {
		case board.MoveUp, board.MoveDown:
			// Moving vertically - check horizontal position relative to walls
			enemyWall := getWallSide(enemy.Head, state.Board.Width, state.Board.Height)

			// Skip if enemy is in corner (risk of self-collision)
			if enemyWall == "corner" {
				continue
			}

			if enemyWall == "left" && myHead.X == 1 {
				// Enemy on left wall, we're one lane inward
				enemyIsOnWall = true
				weAreInward = true
				cutoffMove = board.MoveLeft
			} else if enemyWall == "right" && myHead.X == state.Board.Width-2 {
				// Enemy on right wall, we're one lane inward
				enemyIsOnWall = true
				weAreInward = true
				cutoffMove = board.MoveRight
			}

		case board.MoveLeft, board.MoveRight:
			// Moving horizontally - check vertical position relative to walls
			enemyWall := getWallSide(enemy.Head, state.Board.Width, state.Board.Height)

			// Skip if enemy is in corner (risk of self-collision)
			if enemyWall == "corner" {
				continue
			}

			if enemyWall == "bottom" && myHead.Y == 1 {
				// Enemy on bottom wall, we're one lane inward
				enemyIsOnWall = true
				weAreInward = true
				cutoffMove = board.MoveDown
			} else if enemyWall == "top" && myHead.Y == state.Board.Height-2 {
				// Enemy on top wall, we're one lane inward
				enemyIsOnWall = true
				weAreInward = true
				cutoffMove = board.MoveUp
			}
		}

		// Condition 2: Opponent on wall, we're inward
		if !enemyIsOnWall || !weAreInward {
			continue
		}

		// Condition 3: Our head at least 2 tiles ahead
		var ourAdvance int
		switch myDirection {
		case board.MoveUp:
			ourAdvance = myHead.Y - enemy.Head.Y
		case board.MoveDown:
			ourAdvance = enemy.Head.Y - myHead.Y
		case board.MoveLeft:
			ourAdvance = enemy.Head.X - myHead.X
		case board.MoveRight:
			ourAdvance = myHead.X - enemy.Head.X
		}

		if ourAdvance < 2 {
			continue
		}

		// Condition 4: Our body extends at least 2 tiles past opponent's head
		// This means we need body segments that are at least 2 positions ahead
		maxBodyExtension := 0
		for _, segment := range state.You.Body {
			// Check how far ahead this segment is from enemy head in movement direction
			var segmentAdvance int
			switch myDirection {
			case board.MoveUp:
				segmentAdvance = segment.Y - enemy.Head.Y
			case board.MoveDown:
				segmentAdvance = enemy.Head.Y - segment.Y
			case board.MoveLeft:
				segmentAdvance = enemy.Head.X - segment.X
			case board.MoveRight:
				segmentAdvance = segment.X - enemy.Head.X
			}

			// Track the maximum extension
			if segmentAdvance > maxBodyExtension {
				maxBodyExtension = segmentAdvance
			}
		}

		// Need at least 2 tiles of body past enemy head
		if maxBodyExtension < 2 {
			continue
		}

		// Additional safety: Verify cutoff move won't kill us
		cutoffPos := board.GetNextPosition(myHead, cutoffMove)
		skipTails := true
		if !state.Board.IsInBounds(cutoffPos) || state.Board.IsOccupied(cutoffPos, skipTails) {
			continue
		}

		// Additional safety: Check we're not near a corner in the cutoff direction
		// If we're moving toward a corner, skip to avoid self-collision
		ourWall := getWallSide(myHead, state.Board.Width, state.Board.Height)
		if ourWall == "corner" {
			continue
		}

		// All conditions met - return cutoff kill move!
		return cutoffMove
	}

	return ""
}

// getDirection determines the direction a snake is moving based on head and neck positions
// Returns empty string if direction cannot be determined (snake too short or not moving in cardinal direction)
func getDirection(head, neck board.Coord) string {
	dx := head.X - neck.X
	dy := head.Y - neck.Y

	if dx == 1 && dy == 0 {
		return board.MoveRight
	} else if dx == -1 && dy == 0 {
		return board.MoveLeft
	} else if dy == 1 && dx == 0 {
		return board.MoveUp
	} else if dy == -1 && dx == 0 {
		return board.MoveDown
	}

	return "" // Not moving in a cardinal direction (shouldn't happen in Battlesnake)
}

// getWallSide returns which wall(s) a position is on ("left", "right", "top", "bottom", or "corner")
// Returns empty string if not on any wall
func getWallSide(pos board.Coord, boardWidth, boardHeight int) string {
	onLeft := pos.X == 0
	onRight := pos.X == boardWidth-1
	onTop := pos.Y == boardHeight-1
	onBottom := pos.Y == 0

	wallCount := 0
	if onLeft {
		wallCount++
	}
	if onRight {
		wallCount++
	}
	if onTop {
		wallCount++
	}
	if onBottom {
		wallCount++
	}

	if wallCount >= 2 {
		return "corner"
	} else if onLeft {
		return "left"
	} else if onRight {
		return "right"
	} else if onTop {
		return "top"
	} else if onBottom {
		return "bottom"
	}

	return ""
}

// EvaluateWallInterception detects when an enemy is heading toward a wall and calculates
// an interception route to trap them. Returns bonus score if this move helps with interception.
// NEW AGGRESSIVE TACTIC: Cut off enemies heading to walls
func EvaluateWallInterception(state *board.GameState, nextPos board.Coord) float64 {
	score := 0.0
	myHead := state.You.Head
	
	// Only attempt when we're healthy and aggressive
	if state.You.Health < 50 {
		return 0.0
	}
	
	for _, enemy := range state.Board.Snakes {
		if enemy.ID == state.You.ID || len(enemy.Body) < 2 {
			continue
		}
		
		// Determine enemy's direction
		enemyHead := enemy.Head
		enemyNeck := enemy.Body[1]
		enemyDx := enemyHead.X - enemyNeck.X
		enemyDy := enemyHead.Y - enemyNeck.Y
		
		// Check if enemy is heading toward a wall (within 5 tiles and moving toward it)
		isHeadingToWall := false
		var wallX, wallY int
		
		// Check left wall
		if enemyDx < 0 && enemyHead.X <= 5 {
			isHeadingToWall = true
			wallX = 0
			wallY = enemyHead.Y
		}
		// Check right wall
		if enemyDx > 0 && enemyHead.X >= state.Board.Width-6 {
			isHeadingToWall = true
			wallX = state.Board.Width - 1
			wallY = enemyHead.Y
		}
		// Check bottom wall
		if enemyDy < 0 && enemyHead.Y <= 5 {
			isHeadingToWall = true
			wallX = enemyHead.X
			wallY = 0
		}
		// Check top wall
		if enemyDy > 0 && enemyHead.Y >= state.Board.Height-6 {
			isHeadingToWall = true
			wallX = enemyHead.X
			wallY = state.Board.Height - 1
		}
		
		if !isHeadingToWall {
			continue
		}
		
		// Calculate interception point - ahead of enemy along their path
		interceptX := enemyHead.X + enemyDx * 2
		interceptY := enemyHead.Y + enemyDy * 2
		
		// Clamp to board bounds
		if interceptX < 0 {
			interceptX = 0
		} else if interceptX >= state.Board.Width {
			interceptX = state.Board.Width - 1
		}
		if interceptY < 0 {
			interceptY = 0
		} else if interceptY >= state.Board.Height {
			interceptY = state.Board.Height - 1
		}
		
		interceptPoint := board.Coord{X: interceptX, Y: interceptY}
		
		// Calculate distances
		myDistToIntercept := board.ManhattanDistance(myHead, interceptPoint)
		enemyDistToWall := board.ManhattanDistance(enemyHead, board.Coord{X: wallX, Y: wallY})
		nextDistToIntercept := board.ManhattanDistance(nextPos, interceptPoint)
		
		// Only pursue if we can reasonably intercept (we're closer or equal distance)
		if myDistToIntercept > enemyDistToWall+3 {
			continue
		}
		
		// Check if this move brings us closer to the interception point
		if nextDistToIntercept < myDistToIntercept {
			// Moving toward interception point
			closerBy := float64(myDistToIntercept - nextDistToIntercept)
			
			// Bonus scales with how much closer we get and how trapped the enemy is
			trapFactor := 1.0
			if enemyDistToWall <= 3 {
				trapFactor = 2.0 // Enemy is very close to wall - big opportunity
			}
			
			// Size advantage bonus
			sizeAdvantage := 1.0
			if state.You.Length > enemy.Length+2 {
				sizeAdvantage = 1.5 // We're significantly larger
			} else if state.You.Length <= enemy.Length {
				sizeAdvantage = 0.5 // They're equal or larger - less aggressive
			}
			
			score += closerBy * 30.0 * trapFactor * sizeAdvantage
		}
	}
	
	return score
}

// EvaluateParallelEdgeAvoidance detects when agent and opponent are moving parallel 
// toward the same edge and penalizes moves that reduce lateral separation.
// This prevents the agent from turning toward the opponent when near board boundaries,
// which leads to reduced maneuvering space and higher chance of entrapment.
//
// Returns penalty (negative score) if move reduces distance between parallel lanes near edge.
func EvaluateParallelEdgeAvoidance(state *board.GameState, nextPos board.Coord, move string) float64 {
	penalty := 0.0
	myHead := state.You.Head
	
	// Need at least 2 body segments to determine our direction
	if len(state.You.Body) < 2 {
		return 0.0
	}
	
	myNeck := state.You.Body[1]
	myDirection := getDirection(myHead, myNeck)
	
	if myDirection == "" {
		return 0.0
	}
	
	// Check each enemy snake
	for _, enemy := range state.Board.Snakes {
		if enemy.ID == state.You.ID {
			continue
		}
		
		// Enemy needs at least 2 segments to determine direction
		if len(enemy.Body) < 2 {
			continue
		}
		
		enemyNeck := enemy.Body[1]
		enemyDirection := getDirection(enemy.Head, enemyNeck)
		
		// Condition 1: Both moving in same direction (parallel)
		if myDirection != enemyDirection {
			continue
		}
		
		// Condition 2: Check if we're within EdgeParallelDistance tiles of boundary
		// in the movement direction
		var distToEdge int
		var weAreNearEdge bool
		
		switch myDirection {
		case board.MoveUp:
			// Moving up, check distance to top edge
			distToEdge = state.Board.Height - 1 - myHead.Y
			weAreNearEdge = distToEdge <= EdgeParallelDistance
		case board.MoveDown:
			// Moving down, check distance to bottom edge
			distToEdge = myHead.Y
			weAreNearEdge = distToEdge <= EdgeParallelDistance
		case board.MoveLeft:
			// Moving left, check distance to left edge
			distToEdge = myHead.X
			weAreNearEdge = distToEdge <= EdgeParallelDistance
		case board.MoveRight:
			// Moving right, check distance to right edge
			distToEdge = state.Board.Width - 1 - myHead.X
			weAreNearEdge = distToEdge <= EdgeParallelDistance
		}
		
		if !weAreNearEdge {
			continue
		}
		
		// Condition 3: Check if opponent is in adjacent lane (1 tile offset perpendicular to movement)
		// and ahead or aligned in the movement direction
		var isAdjacentLane bool
		var isAheadOrAligned bool
		var currentLaneOffset int
		
		switch myDirection {
		case board.MoveUp, board.MoveDown:
			// Moving vertically, check horizontal offset (lane separation)
			currentLaneOffset = int(math.Abs(float64(myHead.X - enemy.Head.X)))
			isAdjacentLane = (currentLaneOffset == 1)
			
			// Check if enemy is ahead or aligned in Y direction
			if myDirection == board.MoveUp {
				isAheadOrAligned = enemy.Head.Y >= myHead.Y
			} else {
				isAheadOrAligned = enemy.Head.Y <= myHead.Y
			}
			
		case board.MoveLeft, board.MoveRight:
			// Moving horizontally, check vertical offset (lane separation)
			currentLaneOffset = int(math.Abs(float64(myHead.Y - enemy.Head.Y)))
			isAdjacentLane = (currentLaneOffset == 1)
			
			// Check if enemy is ahead or aligned in X direction
			if myDirection == board.MoveRight {
				isAheadOrAligned = enemy.Head.X >= myHead.X
			} else {
				isAheadOrAligned = enemy.Head.X <= myHead.X
			}
		}
		
		if !isAdjacentLane || !isAheadOrAligned {
			continue
		}
		
		// Condition 4: Check if the move being considered would reduce distance between lanes
		// Calculate what the lane offset would be after this move
		var nextLaneOffset int
		var wouldReduceSeparation bool
		
		switch myDirection {
		case board.MoveUp, board.MoveDown:
			// Moving vertically, check if turning left/right moves toward enemy
			nextLaneOffset = int(math.Abs(float64(nextPos.X - enemy.Head.X)))
			wouldReduceSeparation = nextLaneOffset < currentLaneOffset
			
		case board.MoveLeft, board.MoveRight:
			// Moving horizontally, check if turning up/down moves toward enemy
			nextLaneOffset = int(math.Abs(float64(nextPos.Y - enemy.Head.Y)))
			wouldReduceSeparation = nextLaneOffset < currentLaneOffset
		}
		
		// If move reduces separation when we're near edge and parallel, penalize it
		if wouldReduceSeparation {
			cfg := config.GetConfig()
			
			// Base penalty for reducing separation
			basePenalty := cfg.Tactics.ParallelEdgeBasePenalty
			
			// Increase penalty if very close to edge (1 tile away)
			if distToEdge <= 1 {
				basePenalty = cfg.Tactics.ParallelEdgeClosePenalty
			}
			
			// Increase penalty if enemy is equal or larger (more dangerous)
			if enemy.Length >= state.You.Length {
				basePenalty *= cfg.Tactics.ParallelEdgeLargerMultiplier
			}
			
			penalty += basePenalty
		}
	}
	
	return penalty
}
