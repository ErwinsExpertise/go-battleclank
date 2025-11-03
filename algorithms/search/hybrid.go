package search

import (
	"time"

	"github.com/ErwinsExpertise/go-battleclank/config"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/ErwinsExpertise/go-battleclank/heuristics"
)

// HybridSearch combines multiple search strategies based on game state
type HybridSearch struct {
	greedy                 *GreedySearch
	lookahead              *LookaheadSearch
	mcts                   *MCTSSearch
	useLookaheadOnCritical bool
	lookaheadDepth         int
	useMCTSInEndgame       bool
	mctsIterations         int
	mctsTimeoutMs          int
	criticalHealth         int
	criticalSpaceRatio     float64
	criticalNearbyEnemies  int
	lateGameTurnThreshold  int
}

// NewHybridSearch creates a hybrid search combining multiple strategies using config
func NewHybridSearch() *HybridSearch {
	cfg := config.GetConfig()
	timeoutDuration := time.Duration(cfg.Hybrid.MCTSTimeoutMs) * time.Millisecond
	return &HybridSearch{
		greedy:                 NewGreedySearch(),
		lookahead:              NewLookaheadSearch(cfg.Hybrid.LookaheadDepth),
		mcts:                   NewMCTSSearch(cfg.Hybrid.MCTSIterations, timeoutDuration),
		useLookaheadOnCritical: cfg.Hybrid.UseLookaheadOnCritical,
		lookaheadDepth:         cfg.Hybrid.LookaheadDepth,
		useMCTSInEndgame:       cfg.Hybrid.UseMCTSInEndgame,
		mctsIterations:         cfg.Hybrid.MCTSIterations,
		mctsTimeoutMs:          cfg.Hybrid.MCTSTimeoutMs,
		criticalHealth:         cfg.Hybrid.CriticalHealth,
		criticalSpaceRatio:     cfg.Hybrid.CriticalSpaceRatio,
		criticalNearbyEnemies:  cfg.Hybrid.CriticalNearbyEnemies,
		lateGameTurnThreshold:  cfg.LateGame.TurnThreshold,
	}
}

// FindBestMove selects the best algorithm based on game state
func (h *HybridSearch) FindBestMove(state *board.GameState) string {
	// Determine game phase and criticality
	phase := h.determineGamePhase(state)
	criticality := h.assessCriticality(state)

	// Critical situations: Use deeper search (but fast)
	if criticality == "critical" && h.useLookaheadOnCritical {
		// Use lookahead for critical situations (traps, low health)
		if phase == "early" || phase == "mid" {
			return h.lookahead.FindBestMove(state)
		}
	}

	// Endgame with few snakes: Use MCTS for tactical advantage
	if phase == "late" && len(state.Board.Snakes) <= 2 && h.useMCTSInEndgame {
		// MCTS can find winning sequences in endgame
		move := h.mcts.FindBestMove(state)
		if move != "" {
			return move
		}
	}

	// Default: Use greedy (fast and reliable)
	return h.greedy.FindBestMove(state)
}

// ScoreMove delegates to greedy for scoring
func (h *HybridSearch) ScoreMove(state *board.GameState, move string) float64 {
	return h.greedy.ScoreMove(state, move)
}

// determineGamePhase categorizes the game into early/mid/late
func (h *HybridSearch) determineGamePhase(state *board.GameState) string {
	turn := state.Turn
	snakes := len(state.Board.Snakes)

	// Early game: First 30 turns or 4 snakes
	if turn < 30 || snakes >= 4 {
		return "early"
	}

	// Late game: After turn threshold or 1v1
	if turn > h.lateGameTurnThreshold || snakes == 2 {
		return "late"
	}

	// Mid game: Everything else
	return "mid"
}

// assessCriticality determines how critical the current situation is
func (h *HybridSearch) assessCriticality(state *board.GameState) string {
	you := state.You

	// Critical: Low health
	if you.Health < h.criticalHealth {
		return "critical"
	}

	// Critical: Trapped (low accessible space)
	accessibleSpace := heuristics.FloodFill(state, you.Head, 121)
	spaceRatio := float64(accessibleSpace) / float64(len(you.Body))
	if spaceRatio < h.criticalSpaceRatio {
		return "critical"
	}

	// Critical: Multiple nearby enemies
	nearbyEnemies := 0
	for _, enemy := range state.Board.Snakes {
		if enemy.ID == you.ID {
			continue
		}
		dist := board.ManhattanDistance(you.Head, enemy.Head)
		if dist <= 3 {
			nearbyEnemies++
		}
	}
	if nearbyEnemies >= h.criticalNearbyEnemies {
		return "critical"
	}

	return "normal"
}

// SearchWithInfo returns move and search strategy used (for debugging)
func (h *HybridSearch) SearchWithInfo(state *board.GameState) (string, string) {
	phase := h.determineGamePhase(state)
	criticality := h.assessCriticality(state)

	strategy := "greedy"
	var move string

	if criticality == "critical" {
		if phase == "early" || phase == "mid" {
			strategy = "lookahead"
			move = h.lookahead.FindBestMove(state)
		} else {
			move = h.greedy.FindBestMove(state)
		}
	} else if phase == "late" && len(state.Board.Snakes) <= 2 {
		strategy = "mcts"
		move = h.mcts.FindBestMove(state)
		if move == "" {
			strategy = "greedy-fallback"
			move = h.greedy.FindBestMove(state)
		}
	} else {
		move = h.greedy.FindBestMove(state)
	}

	return move, strategy
}
