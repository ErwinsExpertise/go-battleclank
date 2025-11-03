# go-battleclank

A competitive Battlesnake implementation in Go with advanced decision-making algorithms.
  
## Architecture

### Stateless Design

This Battlesnake is **completely stateless** - all decision-making is based purely on the current game state provided by the Battlesnake API, with no memory of previous turns. This design ensures:

- **Fresh decisions every turn** based on updated board state
- **Easy debugging and testing** without hidden state dependencies
- **Simple horizontal scaling** - any server instance can handle any request
- **No accumulation of bugs** from stale or incorrect state
- **Reliability** - crashes or restarts don't affect decision quality

This follows best practices from top-performing Battlesnakes. See: [Battlesnake Post Mortem](https://medium.com/asymptoticlabs/battlesnake-post-mortem-a5917f9a3428)

## Features

This Battlesnake implements several optimization strategies to maximize survival and winning chances:

### Core Algorithms

1. **Collision Detection**
   - Avoids walls, self-collision, and other snakes
   - Handles tail-chasing optimization (recognizes when tail will move)
   - Detects when snakes just ate (tail won't move)

2. **Flood Fill Space Analysis**
   - Evaluates available space before committing to a move
   - Prevents getting trapped in confined areas
   - Uses recursive algorithm with depth limiting for performance

3. **Food Seeking with Health Management**
   - Always seeks food to prevent starvation and circular behavior
   - Increases food-seeking aggression as health decreases
   - Uses A* pathfinding when health is low (< 50) for accurate navigation
   - Uses Manhattan distance when healthy for better performance
   - **Avoids dangerous food near enemy snakes** to prevent traps
   - Balances food seeking with survival

4. **Head-to-Head Collision Avoidance**
   - Predicts potential head-to-head collisions
   - Avoids conflicts with larger or equal-sized snakes
   - Only engages smaller snakes in head-to-head situations

5. **Strategic Positioning**
   - Prefers center positions early in the game
   - Always actively hunts for food or prey (no circling/standing still)
   - Balances multiple heuristics with weighted scoring

## API Endpoints

- `GET /` - Returns Battlesnake metadata (color, head, tail, etc.)
- `POST /start` - Called when a game starts
- `POST /move` - Called each turn to get the next move
- `POST /end` - Called when a game ends

## Installation

### Download Pre-built Binaries

Download the latest release for your platform from the [releases page](https://github.com/ErwinsExpertise/go-battleclank/releases).

Available platforms:
- Linux (amd64, arm64, armv7)
- macOS (amd64, arm64/Apple Silicon)
- Windows (amd64)

Extract and run:
```bash
./go-battleclank
# Or on Windows: go-battleclank.exe
```

### Using Docker

```bash
docker pull ghcr.io/erwinsexpertise/go-battleclank:latest
docker run -p 8000:8000 ghcr.io/erwinsexpertise/go-battleclank:latest
```

### Building from Source

Prerequisites: Go 1.24 or higher

```bash
git clone https://github.com/ErwinsExpertise/go-battleclank.git
cd go-battleclank
go build
./go-battleclank
```

### Configuration

Set a custom port:
```bash
PORT=8080 ./go-battleclank
```

The server will start on `http://0.0.0.0:8000` by default.

#### GPU Acceleration (Production Ready)

GPU acceleration is available for systems with NVIDIA GPUs and CUDA support:

**Build with CUDA support:**
```bash
go build -tags cuda -o go-battleclank-cuda .
./go-battleclank-cuda --enable-gpu
```

**Build without CUDA (default):**
```bash
go build -o go-battleclank .
./go-battleclank --enable-gpu  # Will use CPU fallback
```

**Requirements for GPU acceleration:**
- NVIDIA GPU with CUDA compute capability 3.0+
- CUDA Toolkit 10.0 or higher
- NVIDIA GPU drivers
- CGO enabled (default in Go)
- **Linux system (Ubuntu, Debian, etc.) - Windows users should use WSL2**

**Benefits with CUDA:**
- 5-10x faster MCTS simulations
- Better move decisions from more complete search
- Estimated +5-10% win rate improvement

**Graceful Fallback:**
The application automatically falls back to CPU if:
- Built without `-tags cuda`
- CUDA not installed on the system
- No NVIDIA GPU detected
- GPU initialization fails

See [BUILD_WITH_CUDA.md](BUILD_WITH_CUDA.md) for detailed build instructions and [GPU_USAGE.md](GPU_USAGE.md) for usage guide.

### Check Version

```bash
./go-battleclank --version
```

## Testing

Run all unit tests:

```bash
go test -v ./...
```

Run with coverage:

```bash
go test -v -cover ./...
```

## Development Setup

For development, testing, and training, use the quick setup script to install all dependencies and build all binaries:

```bash
./tools/quick_setup.sh
```

This script will:
- Check and install Python3 and pip3 (if needed on Ubuntu)
- Verify Go and Rust/Cargo are installed (provides install instructions if missing)
- Install Python dependencies from `requirements.txt`
- Install the battlesnake CLI tool
- Build both the Go snake (`go-battleclank`) and Rust baseline snake
- Make all scripts executable
- Verify all binaries are present

**Prerequisites:**
- Ubuntu 22.04 or 24.04 (other Linux distros should work with minor adjustments)
- Go 1.24 or higher
- Rust/Cargo (script will provide installation instructions if not present)

After setup, you can run:
- `python3 tools/run_benchmark.py 100` - Run live benchmarks against Rust baseline
- `./tools/start_training.sh` - Start continuous training
- `python3 tools/optimize_weights.py` - Optimize decision weights
- `./tools/automated_test_cycle.sh` - Run full test cycle

### Benchmarking

Run performance benchmarks against simulated opponents:

```bash
# Build the benchmark tool
go build -o benchmark_vs_baseline tools/benchmark_vs_baseline.go

# Run 100 games (default)
./benchmark_vs_baseline

# Run custom number of games
./benchmark_vs_baseline 50
```

The benchmark tool provides:
- Win/loss statistics
- Cause of death analysis
- Performance metrics (turns survived, food collected, final length)
- JSON output for further analysis

**Note**: Current benchmark tests against random opponent moves. For testing against the actual Rust baseline snake, both snakes must be running simultaneously and connected via the Battlesnake CLI.

## Project Structure

- `main.go` - Entry point
- `server.go` - HTTP server and handlers
- `models.go` - API data structures
- `logic.go` - Core decision-making logic and algorithms
- `logic_test.go` - Unit tests for logic functions
- `server_test.go` - Unit tests for HTTP handlers

## API Reference

This implementation follows the official Battlesnake API v1:
https://docs.battlesnake.com/api

## Algorithm Details

For detailed algorithm documentation, see:
- **[ALGORITHMS.md](ALGORITHMS.md)** - In-depth technical documentation of current algorithms
- **[STRATEGY_REVIEW.md](STRATEGY_REVIEW.md)** - Comprehensive strategy analysis and future improvements
- **[ASTAR_IMPLEMENTATION.md](ASTAR_IMPLEMENTATION.md)** - Implementation guide for A* pathfinding

### Move Scoring

Each possible move is scored using multiple weighted factors:

- **Space Availability** (weight: 100-200): Amount of reachable space via flood fill
  - Doubled to 200 when enemies are nearby
- **Food Proximity** (dynamic weight): Distance to nearest food
  - Critical health (< 30): weight 400 (350 when outmatched)
  - Low health (< 50): weight 250 (180 when outmatched)
  - Healthy (≥ 50): weight 150 (90 when outmatched) - always aggressively hunting for growth
- **Danger Zone Avoidance** (dynamic penalty): Avoiding positions enemies can reach
  - -700 for significantly larger enemies (2+ length advantage)
  - -400 for similar-sized enemies
  - -100 for smaller enemies
- **Head Collision Risk** (weight: -500): Potential for head-to-head with larger snakes
- **Center Proximity** (weight: 10-15): Distance to board center
  - Early game (turn < 50): weight 10
  - Late game when healthy and not outmatched: weight 15
- **Tail Proximity** (weight: 5): Minimal fallback only when no food exists (prevents circular behavior)
- **Wall Avoidance** (weight: -300): Penalty for positions near walls/corners when enemies present
- **Cutoff Detection** (weight: -300): Penalty for limited escape routes

Fatal moves (out of bounds, snake collision) receive a score of -10000.

### Optimization Techniques

1. **Flood Fill Depth Limiting**: Limits recursion to snake length for performance
2. **Tail Recognition**: Distinguishes between moving and stationary tails
3. **Health-Based Strategy**: Adjusts behavior based on current health level
4. **Multi-Factor Decision Making**: Combines multiple heuristics for robust choices
5. **Always Hunting**: Eliminates circular tail-chasing behavior, snake continuously seeks food or optimal positioning

### Recent Improvements

The refactored engine (now default) includes several enhancements designed to match and exceed baseline opponent strategies:

1. **Ratio-Based Trap Detection**: Uses space-to-body-length ratios (40%, 60%, 80% thresholds) for graduated trap warnings
2. **Food Death Trap Detection**: Checks if eating food would trap the snake (70% threshold since tail doesn't move)
3. **One-Move Lookahead**: Simulates future positions to detect dead-end paths before committing
4. **Enhanced Space Evaluation**: Proper ratio calculations instead of absolute space counts
5. **Survival Priority System**: Clear penalty hierarchy (death -10000, critical trap -600, severe trap -450, etc.)

These improvements are documented in [BASELINE_ANALYSIS.md](BASELINE_ANALYSIS.md).

### Future Enhancements

Potential areas for further improvement:
- **2-Turn Lookahead**: Extended tactical planning
- **BFS Flood Fill**: Iterative approach instead of recursive for consistency
- **Aggressive Pursuit**: Enhanced hunting logic for smaller opponents
- **Machine Learning**: DQN and reinforcement learning approaches
- **Genetic Algorithms**: Automated weight optimization

See [STRATEGY_REVIEW.md](STRATEGY_REVIEW.md) for detailed analysis and implementation roadmap.

## License

MIT

## Author

go-battleclank
