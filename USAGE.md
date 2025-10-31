# Quick Start Guide

## Running Locally

1. **Build the application:**
   ```bash
   go build
   ```

2. **Run the server:**
   ```bash
   ./go-battleclank
   ```

3. **Test it's working:**
   ```bash
   curl http://localhost:8000/
   ```

   You should see JSON output with the snake's configuration.

## Using Docker

1. **Build the Docker image:**
   ```bash
   docker build -t go-battleclank .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 go-battleclank
   ```

## Deploying to Battlesnake

1. Deploy your server to a public URL (e.g., Heroku, Railway, Render, etc.)

2. Go to [play.battlesnake.com](https://play.battlesnake.com)

3. Create a new Battlesnake and enter your server URL

4. Start playing!

## Testing Your Snake

### Manual Testing

Test the info endpoint:
```bash
curl http://localhost:8000/
```

Test a move with sample data:
```bash
curl -X POST http://localhost:8000/move \
  -H "Content-Type: application/json" \
  -d '{
    "game": {"id": "test", "timeout": 500},
    "turn": 1,
    "board": {
      "height": 11,
      "width": 11,
      "food": [{"x": 5, "y": 5}],
      "snakes": [{
        "id": "you",
        "health": 100,
        "body": [{"x": 3, "y": 3}, {"x": 3, "y": 2}],
        "head": {"x": 3, "y": 3}
      }]
    },
    "you": {
      "id": "you",
      "health": 100,
      "body": [{"x": 3, "y": 3}, {"x": 3, "y": 2}],
      "head": {"x": 3, "y": 3}
    }
  }'
```

### Running Tests

Run all tests:
```bash
go test -v ./...
```

Run with coverage:
```bash
go test -v -cover ./...
```

Run specific test:
```bash
go test -v -run TestMove
```

## Environment Variables

- `PORT`: Set the server port (default: 8000)
  ```bash
  PORT=3000 ./go-battleclank
  ```

## Troubleshooting

### Server won't start
- Check if port 8000 is already in use: `lsof -i :8000`
- Try a different port: `PORT=8080 ./go-battleclank`

### Snake makes invalid moves
- Check the logs for error messages
- Verify your board state is correct
- Run unit tests to ensure logic is working: `go test -v`

### Connection timeout
- Ensure your server is publicly accessible
- Check firewall settings
- Verify the URL in Battlesnake dashboard is correct

## Algorithm Tuning

You can adjust the snake's behavior by modifying weights in `logic.go`:

```go
// In scoreMove function
spaceFactor := evaluateSpace(state, nextPos)
score += spaceFactor * 100.0  // Adjust this weight

foodFactor := evaluateFoodProximity(state, nextPos)
score += foodFactor * 200.0   // Adjust this weight

// etc.
```

Adjust health thresholds:
```go
const (
    HealthCritical = 30  // When to prioritize survival
    HealthLow      = 50  // When to seek food aggressively
)
```

## Performance Tips

1. The flood fill is limited to snake length depth for performance
2. All moves are evaluated in parallel conceptually
3. Response time should be under 500ms for most boards
4. Consider profiling if you add more complex algorithms

## Next Steps

- Play games and watch how your snake behaves
- Adjust scoring weights based on performance
- Add more sophisticated algorithms (A*, minimax, etc.)
- Implement game state caching for multi-turn planning
- Add logging to analyze decision patterns
