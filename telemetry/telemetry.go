package telemetry

import (
	"encoding/json"
	"fmt"
	"log"
	"time"
)

// Package telemetry provides logging, metrics, and decision tracking

// MoveDecision records details about a move decision
type MoveDecision struct {
	Timestamp      time.Time              `json:"timestamp"`
	GameID         string                 `json:"game_id"`
	Turn           int                    `json:"turn"`
	ChosenMove     string                 `json:"chosen_move"`
	ChosenScore    float64                `json:"chosen_score"`
	AlternativeMoves map[string]float64   `json:"alternative_moves"`
	Metrics        DecisionMetrics        `json:"metrics"`
	ExecutionTime  time.Duration          `json:"execution_time_ms"`
}

// DecisionMetrics captures key metrics for analysis
type DecisionMetrics struct {
	Health           int     `json:"health"`
	Length           int     `json:"length"`
	AggressionScore  float64 `json:"aggression_score"`
	SpaceAvailable   float64 `json:"space_available"`
	FoodDistance     int     `json:"food_distance"`
	EnemiesNearby    int     `json:"enemies_nearby"`
	TrapOpportunity  bool    `json:"trap_opportunity"`
	InDangerZone     bool    `json:"in_danger_zone"`
}

// GameResult records the outcome of a game
type GameResult struct {
	GameID        string    `json:"game_id"`
	Winner        bool      `json:"winner"`
	FinalTurn     int       `json:"final_turn"`
	FinalLength   int       `json:"final_length"`
	DeathReason   string    `json:"death_reason"` // "collision", "starvation", "head-to-head", etc.
	TotalMoves    int       `json:"total_moves"`
	StartTime     time.Time `json:"start_time"`
	EndTime       time.Time `json:"end_time"`
}

// Logger provides structured logging for telemetry
type Logger struct {
	enabled bool
	verbose bool
}

// NewLogger creates a new telemetry logger
func NewLogger(enabled, verbose bool) *Logger {
	return &Logger{
		enabled: enabled,
		verbose: verbose,
	}
}

// LogMoveDecision logs a move decision
func (l *Logger) LogMoveDecision(decision MoveDecision) {
	if !l.enabled {
		return
	}
	
	if l.verbose {
		data, err := json.MarshalIndent(decision, "", "  ")
		if err != nil {
			log.Printf("Error marshaling move decision: %v", err)
			return
		}
		log.Printf("MOVE_DECISION: %s", string(data))
	} else {
		log.Printf("MOVE %d: %s (score: %.2f, time: %v)", 
			decision.Turn, decision.ChosenMove, decision.ChosenScore, decision.ExecutionTime)
	}
}

// LogGameResult logs a game result
func (l *Logger) LogGameResult(result GameResult) {
	if !l.enabled {
		return
	}
	
	duration := result.EndTime.Sub(result.StartTime)
	
	if l.verbose {
		data, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			log.Printf("Error marshaling game result: %v", err)
			return
		}
		log.Printf("GAME_RESULT: %s", string(data))
	} else {
		status := "LOST"
		if result.Winner {
			status = "WON"
		}
		log.Printf("GAME %s: %s - Turn %d, Length %d, Reason: %s, Duration: %v",
			result.GameID, status, result.FinalTurn, result.FinalLength, 
			result.DeathReason, duration)
	}
}

// LogError logs an error
func (l *Logger) LogError(context string, err error) {
	if !l.enabled {
		return
	}
	log.Printf("ERROR [%s]: %v", context, err)
}

// LogInfo logs informational message
func (l *Logger) LogInfo(message string) {
	if !l.enabled {
		return
	}
	log.Printf("INFO: %s", message)
}

// FormatMoveScores formats move scores for readable output
func FormatMoveScores(scores map[string]float64) string {
	result := ""
	for move, score := range scores {
		result += fmt.Sprintf("%s: %.2f, ", move, score)
	}
	if len(result) > 0 {
		result = result[:len(result)-2] // Remove trailing comma
	}
	return result
}
