package telemetry

import (
	"fmt"
	"sort"
	"time"
)

// FailureAnalysis provides detailed analysis of failure patterns
type FailureAnalysis struct {
	TotalFailures   int
	ByType          map[string]*FailureTypeStats
	ByTurnRange     map[string]int // "early" (0-50), "mid" (51-150), "late" (150+)
	CommonPatterns  []FailurePattern
	Recommendations []string
}

// FailureTypeStats tracks statistics for a specific failure type
type FailureTypeStats struct {
	Count      int
	Percentage float64
	AvgTurn    float64
	AvgHealth  float64
	AvgLength  float64
	Examples   []FailureExample
}

// FailureExample represents a specific failure instance
type FailureExample struct {
	GameID     string
	Turn       int
	Health     int
	Length     int
	BoardState string // Brief description
	LastMove   string
	MoveScore  float64
}

// FailurePattern represents a common failure pattern
type FailurePattern struct {
	Name        string
	Frequency   int
	Description string
	Mitigation  string
}

// FailureAnalyzer analyzes game failures and provides insights
type FailureAnalyzer struct {
	failures []GameResult
}

// NewFailureAnalyzer creates a new failure analyzer
func NewFailureAnalyzer() *FailureAnalyzer {
	return &FailureAnalyzer{
		failures: make([]GameResult, 0),
	}
}

// AddFailure records a game failure
func (fa *FailureAnalyzer) AddFailure(result GameResult) {
	if !result.Winner {
		fa.failures = append(fa.failures, result)
	}
}

// Analyze performs comprehensive failure analysis
func (fa *FailureAnalyzer) Analyze() FailureAnalysis {
	analysis := FailureAnalysis{
		TotalFailures:   len(fa.failures),
		ByType:          make(map[string]*FailureTypeStats),
		ByTurnRange:     make(map[string]int),
		CommonPatterns:  make([]FailurePattern, 0),
		Recommendations: make([]string, 0),
	}

	if len(fa.failures) == 0 {
		return analysis
	}

	// Analyze by failure type
	fa.analyzeByType(&analysis)

	// Analyze by turn range
	fa.analyzeByTurnRange(&analysis)

	// Detect common patterns
	fa.detectPatterns(&analysis)

	// Generate recommendations
	fa.generateRecommendations(&analysis)

	return analysis
}

// analyzeByType breaks down failures by death reason
func (fa *FailureAnalyzer) analyzeByType(analysis *FailureAnalysis) {
	typeCounts := make(map[string][]GameResult)

	// Group failures by type
	for _, failure := range fa.failures {
		typeCounts[failure.DeathReason] = append(typeCounts[failure.DeathReason], failure)
	}

	// Calculate statistics for each type
	for failType, failures := range typeCounts {
		stats := &FailureTypeStats{
			Count:      len(failures),
			Percentage: float64(len(failures)) / float64(analysis.TotalFailures) * 100,
			Examples:   make([]FailureExample, 0),
		}

		totalTurn := 0

		for _, f := range failures {
			totalTurn += f.FinalTurn
		}

		if len(failures) > 0 {
			stats.AvgTurn = float64(totalTurn) / float64(len(failures))
		}

		analysis.ByType[failType] = stats
	}
}

// analyzeByTurnRange categorizes failures by game phase
func (fa *FailureAnalyzer) analyzeByTurnRange(analysis *FailureAnalysis) {
	for _, failure := range fa.failures {
		var rangeKey string
		if failure.FinalTurn <= 50 {
			rangeKey = "early (0-50)"
		} else if failure.FinalTurn <= 150 {
			rangeKey = "mid (51-150)"
		} else {
			rangeKey = "late (150+)"
		}
		analysis.ByTurnRange[rangeKey]++
	}
}

// detectPatterns identifies common failure patterns
func (fa *FailureAnalyzer) detectPatterns(analysis *FailureAnalysis) {
	// Pattern 1: Starvation
	if stats, exists := analysis.ByType["starvation"]; exists && stats.Percentage > 20 {
		analysis.CommonPatterns = append(analysis.CommonPatterns, FailurePattern{
			Name:        "Frequent Starvation",
			Frequency:   stats.Count,
			Description: fmt.Sprintf("%.1f%% of failures are due to starvation", stats.Percentage),
			Mitigation:  "Increase food-seeking priority, especially when health < 30",
		})
	}

	// Pattern 2: Body Collision
	if stats, exists := analysis.ByType["body-collision"]; exists && stats.Percentage > 15 {
		analysis.CommonPatterns = append(analysis.CommonPatterns, FailurePattern{
			Name:        "Self-Trapping",
			Frequency:   stats.Count,
			Description: fmt.Sprintf("%.1f%% of failures are self-collisions", stats.Percentage),
			Mitigation:  "Improve space analysis and escape route planning",
		})
	}

	// Pattern 3: Head-to-Head Losses
	if stats, exists := analysis.ByType["head-to-head-loss"]; exists && stats.Percentage > 10 {
		analysis.CommonPatterns = append(analysis.CommonPatterns, FailurePattern{
			Name:        "Aggressive Head-on Collisions",
			Frequency:   stats.Count,
			Description: fmt.Sprintf("%.1f%% of failures are head-to-head losses", stats.Percentage),
			Mitigation:  "Improve danger zone detection and avoid equal/larger snakes",
		})
	}

	// Pattern 4: Early Game Deaths
	if earlyDeaths, exists := analysis.ByTurnRange["early (0-50)"]; exists {
		earlyPct := float64(earlyDeaths) / float64(analysis.TotalFailures) * 100
		if earlyPct > 30 {
			analysis.CommonPatterns = append(analysis.CommonPatterns, FailurePattern{
				Name:        "Early Game Vulnerability",
				Frequency:   earlyDeaths,
				Description: fmt.Sprintf("%.1f%% of failures occur in first 50 turns", earlyPct),
				Mitigation:  "Improve early-game positioning and initial food acquisition",
			})
		}
	}

	// Pattern 5: Wall Collisions
	if stats, exists := analysis.ByType["wall-collision"]; exists && stats.Percentage > 5 {
		analysis.CommonPatterns = append(analysis.CommonPatterns, FailurePattern{
			Name:        "Wall/Corner Deaths",
			Frequency:   stats.Count,
			Description: fmt.Sprintf("%.1f%% of failures are wall collisions", stats.Percentage),
			Mitigation:  "Increase wall avoidance penalty and corner detection",
		})
	}
}

// generateRecommendations creates actionable recommendations
func (fa *FailureAnalyzer) generateRecommendations(analysis *FailureAnalysis) {
	// Based on top failure types
	sortedTypes := fa.getSortedFailureTypes(analysis)

	if len(sortedTypes) > 0 {
		topFailure := sortedTypes[0]

		switch topFailure {
		case "starvation":
			analysis.Recommendations = append(analysis.Recommendations,
				"Increase food-seeking weight when health < 40",
				"Improve A* pathfinding to avoid blocked food",
				"Add food scarcity detection to prioritize eating earlier",
			)
		case "body-collision", "trapped":
			analysis.Recommendations = append(analysis.Recommendations,
				"Increase space evaluation weight",
				"Improve flood-fill accuracy with better tail handling",
				"Add multi-turn lookahead to detect traps earlier",
			)
		case "head-to-head-loss":
			analysis.Recommendations = append(analysis.Recommendations,
				"Increase danger zone penalty for larger snakes",
				"Improve head-to-head collision prediction",
				"Avoid contested spaces unless we have length advantage",
			)
		case "wall-collision":
			analysis.Recommendations = append(analysis.Recommendations,
				"Increase wall avoidance penalty",
				"Improve corner detection and escape route planning",
				"Add wall proximity to aggression score calculation",
			)
		}
	}

	// General recommendations based on patterns
	if len(analysis.CommonPatterns) >= 3 {
		analysis.Recommendations = append(analysis.Recommendations,
			"Consider implementing MCTS for better multi-turn planning",
			"Add more sophisticated enemy behavior prediction",
			"Tune heuristic weights based on failure statistics",
		)
	}
}

// getSortedFailureTypes returns failure types sorted by frequency
func (fa *FailureAnalyzer) getSortedFailureTypes(analysis *FailureAnalysis) []string {
	type typeCount struct {
		name  string
		count int
	}

	types := make([]typeCount, 0, len(analysis.ByType))
	for name, stats := range analysis.ByType {
		types = append(types, typeCount{name, stats.Count})
	}

	sort.Slice(types, func(i, j int) bool {
		return types[i].count > types[j].count
	})

	sorted := make([]string, len(types))
	for i, tc := range types {
		sorted[i] = tc.name
	}

	return sorted
}

// PrintAnalysis prints a formatted failure analysis report
func PrintAnalysis(analysis FailureAnalysis) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║           FAILURE ANALYSIS REPORT                          ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")

	fmt.Printf("\nTotal Failures Analyzed: %d\n", analysis.TotalFailures)

	if analysis.TotalFailures == 0 {
		fmt.Println("No failures to analyze!")
		return
	}

	// Print failure types
	fmt.Println("\n--- Failure Types ---")

	type typeStats struct {
		name  string
		stats *FailureTypeStats
	}
	sorted := make([]typeStats, 0, len(analysis.ByType))
	for name, stats := range analysis.ByType {
		sorted = append(sorted, typeStats{name, stats})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].stats.Count > sorted[j].stats.Count
	})

	for _, ts := range sorted {
		fmt.Printf("  %-25s: %3d (%.1f%%) - Avg Turn: %.1f\n",
			ts.name, ts.stats.Count, ts.stats.Percentage, ts.stats.AvgTurn)
	}

	// Print turn ranges
	fmt.Println("\n--- Failure by Game Phase ---")
	for phase, count := range analysis.ByTurnRange {
		pct := float64(count) / float64(analysis.TotalFailures) * 100
		fmt.Printf("  %-20s: %3d (%.1f%%)\n", phase, count, pct)
	}

	// Print common patterns
	if len(analysis.CommonPatterns) > 0 {
		fmt.Println("\n--- Common Failure Patterns ---")
		for i, pattern := range analysis.CommonPatterns {
			fmt.Printf("\n%d. %s\n", i+1, pattern.Name)
			fmt.Printf("   Frequency: %d\n", pattern.Frequency)
			fmt.Printf("   Description: %s\n", pattern.Description)
			fmt.Printf("   Mitigation: %s\n", pattern.Mitigation)
		}
	}

	// Print recommendations
	if len(analysis.Recommendations) > 0 {
		fmt.Println("\n--- Recommendations ---")
		for i, rec := range analysis.Recommendations {
			fmt.Printf("%d. %s\n", i+1, rec)
		}
	}

	fmt.Println("\n" + "════════════════════════════════════════════════════════════")
	fmt.Printf("Report generated: %s\n", time.Now().Format("2006-01-02 15:04:05"))
}
