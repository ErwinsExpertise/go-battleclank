# go-battleclank

A competitive Battlesnake implementation in Go with advanced decision-making algorithms.

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
   - Aggressively seeks food when health is low (< 50)
   - Uses Manhattan distance for efficient pathfinding
   - Balances food seeking with survival

4. **Head-to-Head Collision Avoidance**
   - Predicts potential head-to-head collisions
   - Avoids conflicts with larger or equal-sized snakes
   - Only engages smaller snakes in head-to-head situations

5. **Strategic Positioning**
   - Prefers center positions early in the game
   - Follows own tail when health is sufficient
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

- **Space Availability** (weight: 100): Amount of reachable space via flood fill
- **Food Proximity** (weight: 200): Distance to nearest food when health < 50
- **Head Collision Risk** (weight: -500): Potential for head-to-head with larger snakes
- **Center Proximity** (weight: 10): Distance to board center in early game
- **Tail Proximity** (weight: 50): Following own tail when health > 30

Fatal moves (out of bounds, snake collision) receive a score of -10000.

### Optimization Techniques

1. **Flood Fill Depth Limiting**: Limits recursion to snake length for performance
2. **Tail Recognition**: Distinguishes between moving and stationary tails
3. **Health-Based Strategy**: Adjusts behavior based on current health level
4. **Multi-Factor Decision Making**: Combines multiple heuristics for robust choices

### Future Enhancements

The strategy review identifies several potential improvements:
- **A* Pathfinding**: More accurate food seeking around obstacles
- **2-Turn Lookahead**: Tactical planning for better positioning
- **Machine Learning**: DQN and reinforcement learning approaches
- **Genetic Algorithms**: Automated weight optimization

See [STRATEGY_REVIEW.md](STRATEGY_REVIEW.md) for detailed analysis and implementation roadmap.

## License

MIT

## Author

go-battleclank