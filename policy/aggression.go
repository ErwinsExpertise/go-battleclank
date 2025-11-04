package policy

import (
	"github.com/ErwinsExpertise/go-battleclank/config"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
)

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
// INCREASED AGGRESSION: More aggressive when we have size/health advantage
func CalculateAggressionScore(state *board.GameState, mySpace float64) AggressionScore {
	score := 0.5 // Start neutral

	myLength := state.You.Length
	myHealth := state.You.Health

	// Health factor - MORE AGGRESSIVE when healthy
	healthFactor := 0.0
	if myHealth >= AggressionHealthThreshold {
		healthFactor = 0.3 // Was 0.2 - increased by 50%
		score += 0.3
	} else if myHealth < HealthCritical {
		healthFactor = -0.3
		score -= 0.3
	}

	// NEW: Size-based aggression boost
	// After reaching half the board length, become significantly more aggressive
	if state.Board.Width > 0 && state.Board.Height > 0 {
		halfBoardLength := (state.Board.Width + state.Board.Height) / 2
		if myLength >= halfBoardLength {
			score += 0.2
		} else if myLength >= halfBoardLength * 3/4 {
			score += 0.1
		}
	}

	// Length advantage factor - MUCH MORE AGGRESSIVE
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

			// Compare to longest enemy - INCREASED bonuses
			if myLength > longestEnemy+AggressionLengthAdvantage {
				lengthFactor = 0.4 // Was 0.3 - increased by 33%
				score += 0.4
			} else if myLength > longestEnemy {
				lengthFactor = 0.2 // Was 0.1 - doubled
				score += 0.2
			} else if myLength < longestEnemy-AggressionLengthAdvantage {
				lengthFactor = -0.2
				score -= 0.2
			}

			// Compare to average - INCREASED bonus
			if float64(myLength) > avgEnemyLength+1 {
				lengthFactor += 0.15 // Was 0.1 - increased by 50%
				score += 0.15
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
// Uses config values to allow tuning by continuous training
// REDUCED FOOD SEEKING: After reaching half board length, minimize food seeking
func GetFoodWeight(state *board.GameState, aggression AggressionScore, outmatched bool) float64 {
	cfg := getFoodWeightsConfig()
	
	// NEW: Size-based food reduction
	// After reaching half the board length, drastically reduce food seeking
	sizeReductionFactor := 1.0
	if state.Board.Width > 0 && state.Board.Height > 0 {
		halfBoardLength := (state.Board.Width + state.Board.Height) / 2
		
		if state.You.Length >= halfBoardLength {
			// We're big enough - reduce food seeking by 60% to focus on aggression
			sizeReductionFactor = 0.4
		} else if state.You.Length >= halfBoardLength * 3/4 {
			// Getting close - reduce by 30%
			sizeReductionFactor = 0.7
		}
	}

	// Health ceiling - significantly reduce food seeking when very healthy
	if state.You.Health >= cfg.HealthyCeiling {
		// Very healthy - minimal food seeking, focus on positioning
		return cfg.HealthyCeilingWeight * sizeReductionFactor
	}

	if state.You.Health < HealthCritical {
		// Critical health - food is high priority but not absolute
		// Don't apply size reduction when health is critical
		if outmatched {
			return cfg.CriticalHealthOutmatched
		}
		return cfg.CriticalHealth
	} else if state.You.Health < HealthLow {
		// Low health - seek food but consider survival
		// Reduced size penalty when health is low
		reducedFactor := sizeReductionFactor * 0.5 + 0.5 // Soften the reduction
		if outmatched {
			return cfg.LowHealthOutmatched * reducedFactor
		}
		return cfg.LowHealth * reducedFactor
	} else if state.You.Health < 70 {
		// Medium health - moderate food priority
		baseWeight := cfg.MediumHealth
		if outmatched {
			return baseWeight * cfg.MediumHealthOutmatched * sizeReductionFactor
		}
		return baseWeight * sizeReductionFactor
	} else {
		// Healthy (70-79) - reduced food priority
		baseWeight := cfg.HealthyBase * cfg.HealthyMultiplier
		if state.Turn < 50 {
			baseWeight = cfg.HealthyEarlyGame * cfg.HealthyEarlyMultiplier
		}

		if outmatched {
			return baseWeight * cfg.HealthyOutmatched * sizeReductionFactor
		}
		return baseWeight * sizeReductionFactor
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

// FoodWeightsConfig holds food weight configuration
type FoodWeightsConfig struct {
	CriticalHealth           float64
	CriticalHealthOutmatched float64
	LowHealth                float64
	LowHealthOutmatched      float64
	MediumHealth             float64
	MediumHealthOutmatched   float64
	HealthyBase              float64
	HealthyEarlyGame         float64
	HealthyOutmatched        float64
	HealthyCeiling           int
	HealthyCeilingWeight     float64
	HealthyMultiplier        float64
	HealthyEarlyMultiplier   float64
}

// getFoodWeightsConfig returns food weights from config or defaults
func getFoodWeightsConfig() FoodWeightsConfig {
	cfg := config.GetConfig()
	return FoodWeightsConfig{
		CriticalHealth:           cfg.FoodWeights.CriticalHealth,
		CriticalHealthOutmatched: cfg.FoodWeights.CriticalHealthOutmatched,
		LowHealth:                cfg.FoodWeights.LowHealth,
		LowHealthOutmatched:      cfg.FoodWeights.LowHealthOutmatched,
		MediumHealth:             cfg.FoodWeights.MediumHealth,
		MediumHealthOutmatched:   cfg.FoodWeights.MediumHealthOutmatched,
		HealthyBase:              cfg.FoodWeights.HealthyBase,
		HealthyEarlyGame:         cfg.FoodWeights.HealthyEarlyGame,
		HealthyOutmatched:        cfg.FoodWeights.HealthyOutmatched,
		HealthyCeiling:           cfg.FoodWeights.HealthyCeiling,
		HealthyCeilingWeight:     cfg.FoodWeights.HealthyCeilingWeight,
		HealthyMultiplier:        cfg.FoodWeights.HealthyMultiplier,
		HealthyEarlyMultiplier:   cfg.FoodWeights.HealthyEarlyMultiplier,
	}
}
