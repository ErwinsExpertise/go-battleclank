package board

// Package board provides canonical immutable board representation and helper functions
// for the Battlesnake game engine.

// Coord represents a position on the game board
type Coord struct {
	X int `json:"x"`
	Y int `json:"y"`
}

// Snake represents a snake on the board
type Snake struct {
	ID     string  `json:"id"`
	Name   string  `json:"name"`
	Health int     `json:"health"`
	Body   []Coord `json:"body"`
	Head   Coord   `json:"head"`
	Length int     `json:"length"`
}

// Board represents the game board state (immutable)
type Board struct {
	Height  int     `json:"height"`
	Width   int     `json:"width"`
	Food    []Coord `json:"food"`
	Hazards []Coord `json:"hazards"`
	Snakes  []Snake `json:"snakes"`
}

// GameState represents the full state of the game (immutable)
type GameState struct {
	Turn  int   `json:"turn"`
	Board Board `json:"board"`
	You   Snake `json:"you"`
}

// IsInBounds checks if a coordinate is within the board boundaries
func (b *Board) IsInBounds(pos Coord) bool {
	return pos.X >= 0 && pos.X < b.Width && pos.Y >= 0 && pos.Y < b.Height
}

// IsOccupied checks if a position is occupied by any snake body segment
// skipTails indicates whether to consider tails as occupied (tails move unless snake just ate)
func (b *Board) IsOccupied(pos Coord, skipTails bool) bool {
	for _, snake := range b.Snakes {
		for i, segment := range snake.Body {
			// Skip tail if requested and snake hasn't just eaten
			if skipTails && i == len(snake.Body)-1 && snake.Health != 100 {
				continue
			}
			if pos.X == segment.X && pos.Y == segment.Y {
				return true
			}
		}
	}
	return false
}

// GetSnakeByID returns a snake by its ID
func (b *Board) GetSnakeByID(id string) *Snake {
	for i := range b.Snakes {
		if b.Snakes[i].ID == id {
			return &b.Snakes[i]
		}
	}
	return nil
}

// ManhattanDistance calculates Manhattan distance between two coordinates
func ManhattanDistance(a, b Coord) int {
	dx := a.X - b.X
	dy := a.Y - b.Y
	if dx < 0 {
		dx = -dx
	}
	if dy < 0 {
		dy = -dy
	}
	return dx + dy
}

// GetNeighbors returns valid adjacent coordinates (up, down, left, right)
func (b *Board) GetNeighbors(pos Coord) []Coord {
	neighbors := make([]Coord, 0, 4)
	directions := []Coord{
		{X: 0, Y: 1},  // up
		{X: 0, Y: -1}, // down
		{X: -1, Y: 0}, // left
		{X: 1, Y: 0},  // right
	}

	for _, dir := range directions {
		newPos := Coord{X: pos.X + dir.X, Y: pos.Y + dir.Y}
		if b.IsInBounds(newPos) {
			neighbors = append(neighbors, newPos)
		}
	}

	return neighbors
}

// Move directions as constants
const (
	MoveUp    = "up"
	MoveDown  = "down"
	MoveLeft  = "left"
	MoveRight = "right"
)

// AllMoves returns all possible move directions
func AllMoves() []string {
	return []string{MoveUp, MoveDown, MoveLeft, MoveRight}
}

// GetNextPosition returns the coordinate resulting from a move
func GetNextPosition(pos Coord, move string) Coord {
	switch move {
	case MoveUp:
		return Coord{X: pos.X, Y: pos.Y + 1}
	case MoveDown:
		return Coord{X: pos.X, Y: pos.Y - 1}
	case MoveLeft:
		return Coord{X: pos.X - 1, Y: pos.Y}
	case MoveRight:
		return Coord{X: pos.X + 1, Y: pos.Y}
	}
	return pos
}
