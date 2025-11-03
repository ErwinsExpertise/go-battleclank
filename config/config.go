package config

import (
	"log"
	"os"
	"sync"

	"gopkg.in/yaml.v2"
)

// Config holds all battlesnake configuration
type Config struct {
	Search struct {
		Algorithm     string `yaml:"algorithm"`
		MaxDepth      int    `yaml:"max_depth"`
		UseAStar      bool   `yaml:"use_astar"`
		MaxAStarNodes int    `yaml:"max_astar_nodes"`
	} `yaml:"search"`

	Weights struct {
		Space          float64 `yaml:"space"`
		HeadCollision  float64 `yaml:"head_collision"`
		CenterControl  float64 `yaml:"center_control"`
		WallPenalty    float64 `yaml:"wall_penalty"`
		Cutoff         float64 `yaml:"cutoff"`
		Food           float64 `yaml:"food"`
		StraightMovement       float64 `yaml:"straight_movement"`         // bonus for continuing in same direction
		SpaceBaseMultiplier    float64 `yaml:"space_base_multiplier"`    // multiplier for base space weight
		SpaceEnemyMultiplier   float64 `yaml:"space_enemy_multiplier"`   // multiplier when enemies nearby
		SpaceHealthyMultiplier float64 `yaml:"space_healthy_multiplier"` // multiplier when healthy with good space
	} `yaml:"weights"`

	Traps struct {
		Moderate              float64 `yaml:"moderate"`
		Severe                float64 `yaml:"severe"`
		Critical              float64 `yaml:"critical"`
		FoodTrap              float64 `yaml:"food_trap"`              // 50% increase from 800 to prevent dangerous food
		FoodTrapThreshold     float64 `yaml:"food_trap_threshold"`
		FoodTrapCritical      float64 `yaml:"food_trap_critical"`      // penalty at critical health
		FoodTrapLow           float64 `yaml:"food_trap_low"`           // penalty at low health
		SpaceReduction60      float64 `yaml:"space_reduction_60"`      // penalty for 60%+ space reduction
		SpaceReduction50      float64 `yaml:"space_reduction_50"`      // penalty for 50%+ space reduction
		SpaceReductionRatio60 float64 `yaml:"space_reduction_ratio_60"` // threshold ratio for 60% reduction (0.4)
		SpaceReductionRatio50 float64 `yaml:"space_reduction_ratio_50"` // threshold ratio for 50% reduction (0.5)
		SpaceReductionMinBase float64 `yaml:"space_reduction_min_base"` // minimum base space to apply penalties (0.2)
	} `yaml:"traps"`

	Pursuit struct {
		Distance2 float64 `yaml:"distance_2"`
		Distance3 float64 `yaml:"distance_3"`
		Distance4 float64 `yaml:"distance_4"`
		Distance5 float64 `yaml:"distance_5"`
	} `yaml:"pursuit"`

	Trapping struct {
		Weight               float64 `yaml:"weight"`
		SpaceCutoffThreshold float64 `yaml:"space_cutoff_threshold"`
		TrappedRatio         float64 `yaml:"trapped_ratio"`
	} `yaml:"trapping"`

	FoodUrgency struct {
		Critical float64 `yaml:"critical"`
		Low      float64 `yaml:"low"`
		Normal   float64 `yaml:"normal"`
	} `yaml:"food_urgency"`
	
	FoodWeights struct {
		CriticalHealth          float64 `yaml:"critical_health"`
		CriticalHealthOutmatched float64 `yaml:"critical_health_outmatched"`
		LowHealth               float64 `yaml:"low_health"`
		LowHealthOutmatched     float64 `yaml:"low_health_outmatched"`
		MediumHealth            float64 `yaml:"medium_health"`
		MediumHealthOutmatched  float64 `yaml:"medium_health_outmatched"` // multiplier
		HealthyBase             float64 `yaml:"healthy_base"`
		HealthyEarlyGame        float64 `yaml:"healthy_early_game"`
		HealthyOutmatched       float64 `yaml:"healthy_outmatched"` // multiplier
		HealthyCeiling          int     `yaml:"healthy_ceiling"`          // health threshold for minimal food seeking
		HealthyCeilingWeight    float64 `yaml:"healthy_ceiling_weight"`   // food weight when at/above ceiling
		HealthyMultiplier       float64 `yaml:"healthy_multiplier"`       // multiplier for healthy range (70-79)
		HealthyEarlyMultiplier  float64 `yaml:"healthy_early_multiplier"` // multiplier for early game when healthy
	} `yaml:"food_weights"`

	LateGame struct {
		TurnThreshold      int     `yaml:"turn_threshold"`
		CautionMultiplier  float64 `yaml:"caution_multiplier"`
	} `yaml:"late_game"`

	Hybrid struct {
		UseLookaheadOnCritical bool    `yaml:"use_lookahead_on_critical"`
		LookaheadDepth         int     `yaml:"lookahead_depth"`
		UseMCTSInEndgame       bool    `yaml:"use_mcts_in_endgame"`
		MCTSIterations         int     `yaml:"mcts_iterations"`
		MCTSTimeoutMs          int     `yaml:"mcts_timeout_ms"`
		CriticalHealth         int     `yaml:"critical_health"`
		CriticalSpaceRatio     float64 `yaml:"critical_space_ratio"`
		CriticalNearbyEnemies  int     `yaml:"critical_nearby_enemies"`
	} `yaml:"hybrid"`

	Optimization struct {
		Enabled          bool    `yaml:"enabled"`
		LearningRate     float64 `yaml:"learning_rate"`
		BatchSize        int     `yaml:"batch_size"`
		Episodes         int     `yaml:"episodes"`
		ExplorationRate  float64 `yaml:"exploration_rate"`
		DiscountFactor   float64 `yaml:"discount_factor"`
	} `yaml:"optimization"`
}

var (
	globalConfig *Config
	once         sync.Once
	mu           sync.RWMutex
)

// Load reads configuration from file
func Load(filename string) (*Config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var config Config
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, err
	}

	return &config, nil
}

// GetConfig returns the global configuration (singleton)
func GetConfig() *Config {
	once.Do(func() {
		// Try to load from config.yaml, fallback to defaults
		configPath := os.Getenv("BATTLESNAKE_CONFIG")
		if configPath == "" {
			configPath = "config.yaml"
		}

		config, err := Load(configPath)
		if err != nil {
			log.Printf("Warning: Could not load config from %s: %v. Using defaults.", configPath, err)
			config = GetDefaultConfig()
		}

		mu.Lock()
		globalConfig = config
		mu.Unlock()
		
		log.Printf("✓ Configuration loaded successfully:")
		log.Printf("  - Algorithm: %s", config.Search.Algorithm)
		log.Printf("  - Space weight: %.1f", config.Weights.Space)
		log.Printf("  - Head collision weight: %.1f", config.Weights.HeadCollision)
		log.Printf("  - Food trap penalty: %.1f", config.Traps.FoodTrap)
		log.Printf("  - Pursuit distance 2: %.1f", config.Pursuit.Distance2)
		log.Printf("  - Trapping weight: %.1f", config.Trapping.Weight)
	})

	mu.RLock()
	defer mu.RUnlock()
	return globalConfig
}

// ReloadConfig forces a reload of the configuration from file
// This is useful for training when config values are updated
func ReloadConfig() error {
	configPath := os.Getenv("BATTLESNAKE_CONFIG")
	if configPath == "" {
		configPath = "config.yaml"
	}

	config, err := Load(configPath)
	if err != nil {
		return err
	}

	mu.Lock()
	globalConfig = config
	mu.Unlock()
	
	log.Printf("✓ Configuration reloaded:")
	log.Printf("  - Algorithm: %s", config.Search.Algorithm)
	log.Printf("  - Space weight: %.1f", config.Weights.Space)
	log.Printf("  - Head collision weight: %.1f", config.Weights.HeadCollision)
	log.Printf("  - Food trap penalty: %.1f", config.Traps.FoodTrap)
	
	return nil
}

// GetDefaultConfig returns baseline-matched default configuration
func GetDefaultConfig() *Config {
	config := &Config{}

	config.Search.Algorithm = "hybrid"
	config.Search.MaxDepth = 121
	config.Search.UseAStar = true
	config.Search.MaxAStarNodes = 400

	config.Weights.Space = 5.0
	config.Weights.HeadCollision = 500.0
	config.Weights.CenterControl = 2.0
	config.Weights.WallPenalty = 5.0
	config.Weights.Cutoff = 200.0
	config.Weights.Food = 1.0
	config.Weights.StraightMovement = 25.0
	config.Weights.SpaceBaseMultiplier = 1.5
	config.Weights.SpaceEnemyMultiplier = 2.5
	config.Weights.SpaceHealthyMultiplier = 1.2

	config.Traps.Moderate = 250.0
	config.Traps.Severe = 450.0
	config.Traps.Critical = 600.0
	config.Traps.FoodTrap = 1200.0  // Increased 50% from 800 to prevent dangerous food pursuit
	config.Traps.FoodTrapCritical = 500.0
	config.Traps.FoodTrapLow = 800.0
	config.Traps.SpaceReduction60 = 1500.0
	config.Traps.SpaceReduction50 = 800.0
	config.Traps.SpaceReductionRatio60 = 0.4  // 60% reduction threshold
	config.Traps.SpaceReductionRatio50 = 0.5  // 50% reduction threshold
	config.Traps.SpaceReductionMinBase = 0.2  // minimum base space to check
	config.Traps.FoodTrapThreshold = 0.7

	config.Pursuit.Distance2 = 100.0
	config.Pursuit.Distance3 = 50.0
	config.Pursuit.Distance4 = 25.0
	config.Pursuit.Distance5 = 10.0

	config.Trapping.Weight = 400.0
	config.Trapping.SpaceCutoffThreshold = 0.2
	config.Trapping.TrappedRatio = 0.6

	config.FoodUrgency.Critical = 1.8
	config.FoodUrgency.Low = 1.4
	config.FoodUrgency.Normal = 1.0
	
	// Food weight configuration (tunable by training)
	config.FoodWeights.CriticalHealth = 500.0
	config.FoodWeights.CriticalHealthOutmatched = 400.0
	config.FoodWeights.LowHealth = 220.0
	config.FoodWeights.LowHealthOutmatched = 180.0
	config.FoodWeights.MediumHealth = 120.0
	config.FoodWeights.MediumHealthOutmatched = 0.6  // multiplier
	config.FoodWeights.HealthyBase = 80.0
	config.FoodWeights.HealthyEarlyGame = 100.0
	config.FoodWeights.HealthyOutmatched = 0.5  // multiplier
	config.FoodWeights.HealthyCeiling = 80  // health threshold for minimal food seeking
	config.FoodWeights.HealthyCeilingWeight = 10.0  // minimal food weight at/above ceiling
	config.FoodWeights.HealthyMultiplier = 0.5  // multiplier for healthy range (70-79)
	config.FoodWeights.HealthyEarlyMultiplier = 0.6  // multiplier for early game when healthy

	config.LateGame.TurnThreshold = 150
	config.LateGame.CautionMultiplier = 1.1

	config.Hybrid.UseLookaheadOnCritical = true
	config.Hybrid.LookaheadDepth = 3
	config.Hybrid.UseMCTSInEndgame = true
	config.Hybrid.MCTSIterations = 100
	config.Hybrid.MCTSTimeoutMs = 200
	config.Hybrid.CriticalHealth = 30
	config.Hybrid.CriticalSpaceRatio = 3.0
	config.Hybrid.CriticalNearbyEnemies = 2

	config.Optimization.Enabled = false
	config.Optimization.LearningRate = 0.001
	config.Optimization.BatchSize = 32
	config.Optimization.Episodes = 1000
	config.Optimization.ExplorationRate = 0.1
	config.Optimization.DiscountFactor = 0.95

	return config
}

// Save writes configuration to file
func (c *Config) Save(filename string) error {
	data, err := yaml.Marshal(c)
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}
