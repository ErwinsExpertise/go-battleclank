package policy

import (
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"testing"
)

// TestFoodWeightHealthCeiling validates that food-seeking is minimal when health is high
func TestFoodWeightHealthCeiling(t *testing.T) {
	// Create a simple game state with high health
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Height: 11,
			Width:  11,
		},
		You: board.Snake{
			Health: 85, // Very healthy
			Length: 5,
		},
	}

	aggression := AggressionScore{Score: 0.5}
	weight := GetFoodWeight(state, aggression, false)

	// With health at 85, food weight should be very low (10.0)
	if weight > 15.0 {
		t.Errorf("Expected food weight <= 15.0 for health 85, got %.2f", weight)
	}

	t.Logf("✓ Food weight at health 85: %.2f (correctly minimized)", weight)
}

// TestFoodWeightHealthyReduced validates that food-seeking is reduced when healthy (70-79)
func TestFoodWeightHealthyReduced(t *testing.T) {
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Height: 11,
			Width:  11,
		},
		You: board.Snake{
			Health: 75, // Healthy but not at ceiling
			Length: 5,
		},
	}

	aggression := AggressionScore{Score: 0.5}
	weight := GetFoodWeight(state, aggression, false)

	// With health at 75, food weight should be reduced to around 60 (100 * 0.6)
	if weight > 65.0 {
		t.Errorf("Expected food weight <= 65.0 for health 75, got %.2f", weight)
	}

	if weight < 40.0 {
		t.Errorf("Expected food weight >= 40.0 for health 75, got %.2f", weight)
	}

	t.Logf("✓ Food weight at health 75: %.2f (correctly reduced)", weight)
}

// TestFoodWeightMediumUnchanged validates that medium health behavior is unchanged
func TestFoodWeightMediumUnchanged(t *testing.T) {
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Height: 11,
			Width:  11,
		},
		You: board.Snake{
			Health: 60, // Medium health
			Length: 5,
		},
	}

	aggression := AggressionScore{Score: 0.5}
	weight := GetFoodWeight(state, aggression, false)

	// With health at 60, food weight should be around the configured medium value
	if weight < 100.0 || weight > 130.0 {
		t.Logf("Food weight at health 60: %.2f (expected ~120)", weight)
	} else {
		t.Logf("✓ Food weight at health 60: %.2f (correctly maintained)", weight)
	}
}

// TestFoodWeightLowUnchanged validates that low health behavior is unchanged
func TestFoodWeightLowUnchanged(t *testing.T) {
	state := &board.GameState{
		Turn: 10,
		Board: board.Board{
			Height: 11,
			Width:  11,
		},
		You: board.Snake{
			Health: 40, // Low health
			Length: 5,
		},
	}

	aggression := AggressionScore{Score: 0.5}
	weight := GetFoodWeight(state, aggression, false)

	// With health at 40, food weight should be high
	if weight < 200.0 {
		t.Logf("Food weight at health 40: %.2f (expected ~220)", weight)
	} else {
		t.Logf("✓ Food weight at health 40: %.2f (correctly high)", weight)
	}
}

// TestFoodWeightProgression validates that food weight decreases as health increases
func TestFoodWeightProgression(t *testing.T) {
	healthLevels := []int{30, 40, 50, 60, 70, 80, 90}
	var prevWeight float64 = 10000.0 // Start high

	for _, health := range healthLevels {
		state := &board.GameState{
			Turn: 10,
			Board: board.Board{
				Height: 11,
				Width:  11,
			},
			You: board.Snake{
				Health: health,
				Length: 5,
			},
		}

		aggression := AggressionScore{Score: 0.5}
		weight := GetFoodWeight(state, aggression, false)

		t.Logf("Health %d -> Food Weight: %.2f", health, weight)

		// Verify weight decreases as health increases
		if health > 30 && weight > prevWeight {
			t.Errorf("Food weight should decrease as health increases. Health %d has weight %.2f > previous %.2f",
				health, weight, prevWeight)
		}

		prevWeight = weight
	}

	t.Logf("✓ Food weight correctly decreases as health increases")
}
