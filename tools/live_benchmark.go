//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"time"
)

// Battlesnake API structures
type Coord struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type Battlesnake struct {
	ID      string  `json:"id"`
	Name    string  `json:"name"`
	Health  int     `json:"health"`
	Body    []Coord `json:"body"`
	Head    Coord   `json:"head"`
	Length  int     `json:"length"`
	Latency string  `json:"latency"`
	Shout   string  `json:"shout"`
}

type Board struct {
	Height  int           `json:"height"`
	Width   int           `json:"width"`
	Food    []Coord       `json:"food"`
	Snakes  []Battlesnake `json:"snakes"`
	Hazards []Coord       `json:"hazards"`
}

type Game struct {
	ID      string `json:"id"`
	Timeout int    `json:"timeout"`
}

type GameState struct {
	Game  Game        `json:"game"`
	Turn  int         `json:"turn"`
	Board Board       `json:"board"`
	You   Battlesnake `json:"you"`
}

type MoveResponse struct {
	Move  string `json:"move"`
	Shout string `json:"shout,omitempty"`
}

// Game configuration
type GameConfig struct {
	Width    int
	Height   int
	MaxTurns int
	Timeout  int
}

// Game result
type GameResult struct {
	Winner       string
	Turns        int
	GoLength     int
	RustLength   int
	DeathCause   string
	GoSurvived   bool
	RustSurvived bool
}

// Snake client
type SnakeClient struct {
	Name string
	URL  string
}

func (c *SnakeClient) GetMove(state GameState) (string, error) {
	jsonData, err := json.Marshal(state)
	if err != nil {
		return "", err
	}

	client := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := client.Post(c.URL+"/move", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	var moveResp MoveResponse
	if err := json.Unmarshal(body, &moveResp); err != nil {
		return "", err
	}

	return moveResp.Move, nil
}

// Initialize game board
func initializeBoard(config GameConfig) Board {
	board := Board{
		Width:   config.Width,
		Height:  config.Height,
		Food:    make([]Coord, 0),
		Snakes:  make([]Battlesnake, 0),
		Hazards: make([]Coord, 0),
	}

	// Place initial food
	for i := 0; i < 3; i++ {
		board.Food = append(board.Food, Coord{
			X: rand.Intn(config.Width),
			Y: rand.Intn(config.Height),
		})
	}

	return board
}

// Place snakes on board
func placeSnakes(board *Board, goSnake, rustSnake SnakeClient) {
	// Place go snake (left side)
	goStart := Coord{X: 1, Y: board.Height / 2}
	board.Snakes = append(board.Snakes, Battlesnake{
		ID:     "go-snake-id",
		Name:   goSnake.Name,
		Health: 100,
		Body:   []Coord{goStart, goStart, goStart},
		Head:   goStart,
		Length: 3,
	})

	// Place rust snake (right side)
	rustStart := Coord{X: board.Width - 2, Y: board.Height / 2}
	board.Snakes = append(board.Snakes, Battlesnake{
		ID:     "rust-snake-id",
		Name:   rustSnake.Name,
		Health: 100,
		Body:   []Coord{rustStart, rustStart, rustStart},
		Head:   rustStart,
		Length: 3,
	})
}

// Move snake
func moveSnake(snake *Battlesnake, move string, ateFood bool) error {
	head := snake.Head
	var newHead Coord

	switch move {
	case "up":
		newHead = Coord{X: head.X, Y: head.Y + 1}
	case "down":
		newHead = Coord{X: head.X, Y: head.Y - 1}
	case "left":
		newHead = Coord{X: head.X - 1, Y: head.Y}
	case "right":
		newHead = Coord{X: head.X + 1, Y: head.Y}
	default:
		return fmt.Errorf("invalid move: %s", move)
	}

	// Add new head
	snake.Body = append([]Coord{newHead}, snake.Body...)
	snake.Head = newHead

	// Remove tail if didn't eat food
	if !ateFood {
		snake.Body = snake.Body[:len(snake.Body)-1]
	} else {
		snake.Length++
	}

	// Decrease health
	snake.Health--
	if ateFood {
		snake.Health = 100
	}

	return nil
}

// Check collisions and update board
func checkCollisions(board *Board, config GameConfig) {
	aliveSnakes := make([]Battlesnake, 0)

	for i := range board.Snakes {
		snake := &board.Snakes[i]
		if snake.Health <= 0 {
			continue // Already dead
		}

		head := snake.Head

		// Check wall collision
		if head.X < 0 || head.X >= config.Width || head.Y < 0 || head.Y >= config.Height {
			continue
		}

		// Check self collision
		selfCollision := false
		for j, segment := range snake.Body[1:] {
			if head.X == segment.X && head.Y == segment.Y {
				// Check if it's the tail that will move
				if j == len(snake.Body)-2 {
					// Tail will move, not a collision
					continue
				}
				selfCollision = true
				break
			}
		}
		if selfCollision {
			continue
		}

		// Check collision with other snakes
		collision := false
		for j := range board.Snakes {
			if i == j {
				continue
			}
			other := &board.Snakes[j]
			for _, segment := range other.Body {
				if head.X == segment.X && head.Y == segment.Y {
					collision = true
					break
				}
			}
			if collision {
				break
			}
		}
		if collision {
			continue
		}

		aliveSnakes = append(aliveSnakes, *snake)
	}

	board.Snakes = aliveSnakes
}

// Run a single game
func runGame(goSnake, rustSnake SnakeClient, config GameConfig, gameNum int) GameResult {
	board := initializeBoard(config)
	placeSnakes(&board, goSnake, rustSnake)

	game := Game{
		ID:      fmt.Sprintf("game-%d", gameNum),
		Timeout: config.Timeout,
	}

	for turn := 0; turn < config.MaxTurns; turn++ {
		if len(board.Snakes) == 0 {
			return GameResult{
				Winner:     "draw",
				Turns:      turn,
				DeathCause: "all-eliminated",
			}
		}

		if len(board.Snakes) == 1 {
			winner := board.Snakes[0].Name
			goSurvived := winner == goSnake.Name
			rustSurvived := winner == rustSnake.Name

			return GameResult{
				Winner:       winner,
				Turns:        turn,
				GoLength:     getSnakeLength(&board, goSnake.Name),
				RustLength:   getSnakeLength(&board, rustSnake.Name),
				GoSurvived:   goSurvived,
				RustSurvived: rustSurvived,
				DeathCause:   "eliminated",
			}
		}

		// Get moves from both snakes
		moves := make(map[string]string)
		for i := range board.Snakes {
			snake := &board.Snakes[i]
			state := GameState{
				Game:  game,
				Turn:  turn,
				Board: board,
				You:   *snake,
			}

			var client *SnakeClient
			if snake.Name == goSnake.Name {
				client = &goSnake
			} else {
				client = &rustSnake
			}

			move, err := client.GetMove(state)
			if err != nil {
				move = "up" // Default move on error
			}
			moves[snake.ID] = move
		}

		// Move all snakes
		for i := range board.Snakes {
			snake := &board.Snakes[i]
			move := moves[snake.ID]

			// Check if snake ate food
			head := snake.Head
			var newHead Coord
			switch move {
			case "up":
				newHead = Coord{X: head.X, Y: head.Y + 1}
			case "down":
				newHead = Coord{X: head.X, Y: head.Y - 1}
			case "left":
				newHead = Coord{X: head.X - 1, Y: head.Y}
			case "right":
				newHead = Coord{X: head.X + 1, Y: head.Y}
			}

			ateFood := false
			for j, food := range board.Food {
				if newHead.X == food.X && newHead.Y == food.Y {
					ateFood = true
					// Remove food
					board.Food = append(board.Food[:j], board.Food[j+1:]...)
					// Spawn new food
					board.Food = append(board.Food, Coord{
						X: rand.Intn(config.Width),
						Y: rand.Intn(config.Height),
					})
					break
				}
			}

			moveSnake(snake, move, ateFood)
		}

		// Check collisions
		checkCollisions(&board, config)
	}

	// Max turns reached
	if len(board.Snakes) == 0 {
		return GameResult{
			Winner:     "draw",
			Turns:      config.MaxTurns,
			DeathCause: "timeout-all-dead",
		}
	}

	// Find longest snake
	longest := &board.Snakes[0]
	for i := range board.Snakes {
		if board.Snakes[i].Length > longest.Length {
			longest = &board.Snakes[i]
		}
	}

	return GameResult{
		Winner:       longest.Name,
		Turns:        config.MaxTurns,
		GoLength:     getSnakeLength(&board, goSnake.Name),
		RustLength:   getSnakeLength(&board, rustSnake.Name),
		GoSurvived:   isSnakeAlive(&board, goSnake.Name),
		RustSurvived: isSnakeAlive(&board, rustSnake.Name),
		DeathCause:   "timeout",
	}
}

func getSnakeLength(board *Board, name string) int {
	for _, snake := range board.Snakes {
		if snake.Name == name {
			return snake.Length
		}
	}
	return 0
}

func isSnakeAlive(board *Board, name string) bool {
	for _, snake := range board.Snakes {
		if snake.Name == name {
			return true
		}
	}
	return false
}

func main() {
	numGames := flag.Int("games", 100, "Number of games to run")
	goURL := flag.String("go-url", "http://localhost:8000", "Go snake URL")
	rustURL := flag.String("rust-url", "http://localhost:8080", "Rust baseline URL")
	flag.Parse()

	config := GameConfig{
		Width:    11,
		Height:   11,
		MaxTurns: 500,
		Timeout:  500,
	}

	goSnake := SnakeClient{Name: "go-battleclank", URL: *goURL}
	rustSnake := SnakeClient{Name: "rust-baseline", URL: *rustURL}

	fmt.Println("================================================")
	fmt.Println("  Live Battlesnake Benchmark")
	fmt.Println("================================================")
	fmt.Printf("Games: %d\n", *numGames)
	fmt.Printf("Go snake: %s\n", *goURL)
	fmt.Printf("Rust baseline: %s\n\n", *rustURL)

	wins := 0
	losses := 0
	draws := 0

	for i := 1; i <= *numGames; i++ {
		if i%10 == 0 || i == 1 {
			fmt.Printf("Progress: %d/%d games (%.1f%%)\n", i, *numGames, float64(i)/float64(*numGames)*100)
		}

		result := runGame(goSnake, rustSnake, config, i)

		switch result.Winner {
		case goSnake.Name:
			wins++
		case rustSnake.Name:
			losses++
		default:
			draws++
		}
	}

	winRate := float64(wins) / float64(*numGames) * 100

	fmt.Println("\n================================================")
	fmt.Println("  Results Summary")
	fmt.Println("================================================")
	fmt.Printf("Wins:   %d (%.1f%%)\n", wins, winRate)
	fmt.Printf("Losses: %d (%.1f%%)\n", losses, float64(losses)/float64(*numGames)*100)
	fmt.Printf("Draws:  %d (%.1f%%)\n\n", draws, float64(draws)/float64(*numGames)*100)

	if winRate >= 80.0 {
		fmt.Println("✓ SUCCESS: Win rate >= 80% target!")
		os.Exit(0)
	} else if winRate >= 60.0 {
		fmt.Println("⚠ PARTIAL: Win rate >= 60%")
		os.Exit(0)
	} else {
		fmt.Println("✗ BELOW TARGET: Win rate < 60%")
		os.Exit(1)
	}
}
