package heuristics

import (
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
)

// EvaluateInwardTrap attempts to trap enemy in center by surrounding from outside
// Only activates when enemy length exceeds threshold (self-collision risk)
// Returns bonus score if this move helps create an inward trap
func EvaluateInwardTrap(state *board.GameState, nextPos board.Coord) float64 {
	score := 0.0
	centerX := state.Board.Width / 2
	centerY := state.Board.Height / 2
	
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
			if state.Board.IsInBounds(nextPos) && !state.Board.IsOccupied(nextPos, true) {
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
