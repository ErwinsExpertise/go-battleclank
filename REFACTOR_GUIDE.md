# Refactored Architecture Guide

This guide explains the new modular architecture implemented for go-battleclank.

## Overview

The refactored codebase follows a clean, modular architecture inspired by community best practices and proven Battlesnake algorithms. The new structure separates concerns and makes the codebase easier to test, maintain, and extend.

## Architecture

### Module Structure

```
go-battleclank/
├── engine/
│   ├── board/          # Canonical immutable board representation
│   └── simulation/     # Fast game-state simulation for lookahead
├── algorithms/
│   └── search/         # Pluggable search strategies
│       ├── greedy.go   # Single-turn heuristic search
│       └── lookahead.go # Multi-turn lookahead search
├── heuristics/         # Reusable heuristic functions
│   ├── space.go        # Flood-fill space analysis
│   ├── danger.go       # Danger zone prediction
│   ├── food.go         # Food seeking with A* pathfinding
│   └── trap.go         # Trap detection and evaluation
├── policy/             # Decision-making policies
│   └── aggression.go   # Aggression scoring & risk/reward
├── telemetry/          # Logging and metrics
│   └── telemetry.go    # Move decision tracking
└── tests/
    └── sims/           # Test harness for simulations
        ├── harness.go
        └── harness_test.go
```

## Key Components

### 1. Engine (board & simulation)

**engine/board** provides the canonical board representation:
- Immutable data structures
- Coordinate helpers
- Movement functions
- Board state queries

**engine/simulation** handles game state simulation:
- Fast state copying for lookahead
- Move simulation
- Validity checking

### 2. Heuristics

Modular, reusable heuristic functions:

**Space Analysis** (`heuristics/space.go`):
- Flood-fill algorithm for reachable space
- Space comparison between positions
- Snake-specific space calculation

**Danger Zones** (`heuristics/danger.go`):
- Enemy move prediction
- Head-to-head collision detection
- Cutoff/boxing-in detection
- Chase detection

**Food Seeking** (`heuristics/food.go`):
- Manhattan distance calculation
- A* pathfinding for accurate navigation
- Food danger detection (near enemies)

**Trap Detection** (`heuristics/trap.go`):
- Opponent space reduction analysis
- Trap safety validation
- Trap opportunity scoring

### 3. Policy

**Aggression Scoring** (`policy/aggression.go`):
- Calculates aggression score (0.0-1.0) based on:
  - Health level
  - Length advantage
  - Space control
  - Board position
- Decision thresholds for offensive vs defensive play
- Dynamic food weight calculation

### 4. Search Algorithms

**Greedy Search** (`algorithms/search/greedy.go`):
- Single-turn heuristic evaluation
- Combines all heuristics with weighted scoring
- Fast and reliable

**Lookahead Search** (`algorithms/search/lookahead.go`):
- Multi-turn lookahead (configurable depth)
- Simplified MaxN approach
- Enemy response prediction
- Better tactical planning

### 5. Telemetry

**Decision Tracking** (`telemetry/telemetry.go`):
- Structured logging of move decisions
- Score breakdowns for analysis
- Game result tracking
- Performance metrics

### 6. Test Harness

**Simulation Framework** (`tests/sims/harness.go`):
- Scenario-based testing
- Strategy comparison
- Automated validation
- Performance benchmarking

## Usage

### Using the Greedy Strategy

```go
import (
    "github.com/ErwinsExpertise/go-battleclank/algorithms/search"
    "github.com/ErwinsExpertise/go-battleclank/engine/board"
)

func makeMove(state *board.GameState) string {
    strategy := search.NewGreedySearch()
    return strategy.FindBestMove(state)
}
```

### Using the Lookahead Strategy

```go
import (
    "github.com/ErwinsExpertise/go-battleclank/algorithms/search"
    "github.com/ErwinsExpertise/go-battleclank/engine/board"
)

func makeMove(state *board.GameState) string {
    // 2-turn lookahead
    strategy := search.NewLookaheadSearch(2)
    return strategy.FindBestMove(state)
}
```

### Creating Custom Test Scenarios

```go
import "github.com/ErwinsExpertise/go-battleclank/tests/sims"

func runCustomTests() {
    harness := sims.NewTestHarness(true) // true = use greedy
    
    scenario := sims.TestScenario{
        Name: "My custom test",
        State: createTestState(),
        ExpectedMove: "up",
        ShouldSurvive: true,
    }
    
    harness.AddScenario(scenario)
    results := harness.RunAll()
    sims.PrintResults(results)
}
```

## Key Features

### 1. Stateless Design

All functions are pure or depend only on the provided GameState. This ensures:
- Fresh decisions every turn
- Easy testing and debugging
- Simple horizontal scaling
- No state accumulation bugs

### 2. Modular Heuristics

Each heuristic is:
- Self-contained
- Independently testable
- Reusable across strategies
- Easy to tune or replace

### 3. Pluggable Strategies

Different search strategies can be easily swapped:
- Greedy (fast, reliable)
- Lookahead (tactical, slower)
- Custom implementations

### 4. Comprehensive Testing

The architecture supports:
- Unit tests for each module
- Integration tests for strategies
- Scenario-based testing
- Performance benchmarking

## Performance Considerations

### Time Budget (500ms timeout)

**Greedy Strategy**:
- Typical execution: 20-50ms
- Safe margin: ~450ms available
- Recommended for production

**Lookahead Strategy** (depth 2):
- Typical execution: 80-200ms
- Safe margin: ~300ms available
- Use for tactical advantage

**Optimization Tips**:
1. Limit flood-fill depth (currently snake length)
2. Limit A* nodes explored (default: 200)
3. Use greedy fallback in lookahead
4. Profile hotspots before optimizing

### Memory Usage

- Current: ~10-15MB
- Lookahead adds: ~5-10MB
- Well within typical limits

## Extending the Architecture

### Adding New Heuristics

Create a new file in `heuristics/`:

```go
package heuristics

import "github.com/ErwinsExpertise/go-battleclank/engine/board"

func MyNewHeuristic(state *board.GameState, pos board.Coord) float64 {
    // Your logic here
    return score
}
```

### Adding New Search Strategies

Create a new file in `algorithms/search/`:

```go
package search

import "github.com/ErwinsExpertise/go-battleclank/engine/board"

type MyStrategy struct {
    // Configuration
}

func (s *MyStrategy) FindBestMove(state *board.GameState) string {
    // Your logic here
    return bestMove
}

func (s *MyStrategy) ScoreMove(state *board.GameState, move string) float64 {
    // Your scoring logic
    return score
}
```

### Customizing Weights

Modify the weights in `algorithms/search/greedy.go`:

```go
func NewGreedySearch() *GreedySearch {
    return &GreedySearch{
        SpaceWeight:         100.0,  // Increase for more defensive play
        HeadCollisionWeight: 500.0,  // Increase to avoid combat more
        FoodWeight:          200.0,  // Increase for more food seeking
        // ...
    }
}
```

## Migration from Legacy Code

The legacy code in `logic.go` remains functional. To migrate:

1. **Keep legacy code** for backward compatibility
2. **Test new code** using the test harness
3. **Compare results** between old and new implementations
4. **Switch gradually** by routing to new code in production
5. **Monitor telemetry** for issues
6. **Remove legacy** when confident in new implementation

Example integration:

```go
func move(state GameState) BattlesnakeMoveResponse {
    // Convert to internal representation
    internalState := convertToInternalState(state)
    
    // Use new refactored logic
    strategy := search.NewGreedySearch()
    bestMove := strategy.FindBestMove(internalState)
    
    return BattlesnakeMoveResponse{Move: bestMove}
}
```

## Testing Strategy

### Unit Tests

Run module-specific tests:
```bash
go test ./engine/board
go test ./heuristics
go test ./policy
```

### Integration Tests

Run strategy tests:
```bash
go test ./tests/sims
```

### Full Test Suite

Run all tests:
```bash
go test ./...
```

### Benchmarking

```bash
go test -bench=. ./algorithms/search
```

## Future Enhancements

Potential additions to the architecture:

1. **Advanced Search**:
   - Full MaxN implementation
   - MCTS (Monte Carlo Tree Search)
   - Alpha-beta pruning

2. **Machine Learning**:
   - Neural network integration
   - Behavioral cloning
   - Reinforcement learning

3. **Enhanced Telemetry**:
   - Real-time metrics dashboard
   - Failure analysis tools
   - Replay system

4. **Optimization**:
   - Bitboard representations
   - Parallel move evaluation
   - Caching/memoization

## References

- [Battlesnake Docs](https://docs.battlesnake.com)
- [Community Best Practices](https://github.com/BattlesnakeOfficial/community)
- [Useful Algorithms Guide](https://docs.battlesnake.com/guides/useful-algorithms)
- [Original Strategy Review](STRATEGY_REVIEW.md)

## Support

For questions or issues with the refactored architecture:
1. Check existing tests in `tests/sims/`
2. Review module documentation in code
3. Consult the original strategy documents
4. Open an issue on GitHub
