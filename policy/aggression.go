package policy

import "github.com/ErwinsExpertise/go-battleclank/engine/board"

// Package policy provides aggression scoring and risk/reward decision making

const (
	// AggressionLengthAdvantage for aggressive behavior
	AggressionLengthAdvantage = 2
	
	// AggressionHealthThreshold for aggressive behavior
	AggressionHealthThreshold = 60
	
	// Health thresholds
	HealthCritical = 30
	HealthLow      = 50
)

// AggressionScore represents the calculated aggression level
type AggressionScore struct {
	Score          float64 // 0.0 (defensive) to 1.0 (aggressive)
	HealthFactor   float64
	LengthFactor   float64
	SpaceFactor    float64
	PositionFactor float64
}

// CalculateAggressionScore determines how aggressive the snake should be
// Returns a score from 0.0 (defensive) to 1.0 (aggressive)
func CalculateAggressionScore(state *board.GameState, mySpace float64) AggressionScore {
	score := 0.5 // Start neutral
	
	myLength := state.You.Length
	myHealth := state.You.Health
	
	// Health factor
	healthFactor := 0.0
	if myHealth >= AggressionHealthThreshold {
		healthFactor = 0.2
		score += 0.2
	} else if myHealth < HealthCritical {
		healthFactor = -0.3
		score -= 0.3
	}
	
	// Length advantage factor
	lengthFactor := 0.0
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
			
			// Compare to longest enemy
			if myLength > longestEnemy+AggressionLengthAdvantage {
				lengthFactor = 0.3
				score += 0.3
			} else if myLength > longestEnemy {
				lengthFactor = 0.1
				score += 0.1
			} else if myLength < longestEnemy-AggressionLengthAdvantage {
				lengthFactor = -0.2
				score -= 0.2
			}
			
			// Compare to average
			if float64(myLength) > avgEnemyLength+1 {
				lengthFactor += 0.1
				score += 0.1
			}
		}
	}
	
	// Space control factor
	spaceFactor := 0.0
	if mySpace > 0.4 {
		spaceFactor = 0.1
		score += 0.1
	} else if mySpace < 0.2 {
		spaceFactor = -0.2
		score -= 0.2
	}
	
	// Position factor (distance from walls)
	positionFactor := 0.0
	distToWall := getMinDistanceToWall(state, state.You.Head)
	if distToWall <= 1 {
		positionFactor = -0.1
		score -= 0.1
	}
	
	// Clamp between 0 and 1
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}
	
	return AggressionScore{
		Score:          score,
		HealthFactor:   healthFactor,
		LengthFactor:   lengthFactor,
		SpaceFactor:    spaceFactor,
		PositionFactor: positionFactor,
	}
}

// getMinDistanceToWall returns minimum distance to any wall
func getMinDistanceToWall(state *board.GameState, pos board.Coord) int {
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

// ShouldAttemptTrap determines if we should try to trap an enemy
func ShouldAttemptTrap(aggression AggressionScore) bool {
	return aggression.Score > 0.6
}

// ShouldPrioritizeSurvival determines if we should focus on survival
func ShouldPrioritizeSurvival(aggression AggressionScore) bool {
	return aggression.Score < 0.4
}

// GetFoodWeight returns appropriate food seeking weight based on health and aggression
func GetFoodWeight(state *board.GameState, aggression AggressionScore, outmatched bool) float64 {
	if state.You.Health < HealthCritical {
		if outmatched {
			return 350.0
		}
		return 400.0
	} else if state.You.Health < HealthLow {
		if outmatched {
			return 180.0
		}
		return 250.0
	} else {
		// Always seek food when healthy to maintain growth
		if outmatched {
			return 90.0
		}
		return 150.0
	}
}

// IsOutmatched checks if we're significantly outmatched by nearby enemies
func IsOutmatched(state *board.GameState, proximityRadius int) bool {
	myHead := state.You.Head
	myLength := state.You.Length
	
	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}
		
		dist := board.ManhattanDistance(myHead, snake.Head)
		if dist <= proximityRadius {
			// Enemy nearby - check if much larger
			if snake.Length > myLength+4 {
				return true
			}
		}
	}
	
	return false
}
