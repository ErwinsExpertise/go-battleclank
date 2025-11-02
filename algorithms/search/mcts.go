package search

import (
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/ErwinsExpertise/go-battleclank/engine/simulation"
	"github.com/ErwinsExpertise/go-battleclank/gpu"
	"github.com/ErwinsExpertise/go-battleclank/heuristics"
	"log"
	"math"
	"math/rand"
	"time"
)

// MCTSSearch implements Monte Carlo Tree Search for multi-agent decision making
type MCTSSearch struct {
	MaxIterations  int
	MaxTime        time.Duration
	ExplorationC   float64 // UCB exploration constant (typically √2)
	MaxDepth       int
	rng            *rand.Rand
	baseStrategy   *GreedySearch
}

// NewMCTSSearch creates a new MCTS search
func NewMCTSSearch(maxIterations int, maxTime time.Duration) *MCTSSearch {
	return &MCTSSearch{
		MaxIterations: maxIterations,
		MaxTime:       maxTime,
		ExplorationC:  math.Sqrt(2),
		MaxDepth:      15,
		rng:           rand.New(rand.NewSource(time.Now().UnixNano())),
		baseStrategy:  NewGreedySearch(),
	}
}

// MCTSNode represents a node in the MCTS tree
type MCTSNode struct {
	state       *board.GameState
	move        string
	parent      *MCTSNode
	children    []*MCTSNode
	visits      int
	wins        float64
	untriedMoves []string
}

// FindBestMove uses MCTS to find the best move
func (m *MCTSSearch) FindBestMove(state *board.GameState) string {
	// Try GPU-accelerated MCTS first if available
	if gpu.IsAvailable() {
		gpuMCTS := gpu.NewMCTSBatchGPU(100, m.MaxDepth)
		move, err := gpuMCTS.SimulateBatch(state, m.MaxIterations)
		if err == nil {
			return move
		}
		// Fallback to CPU if GPU fails
		log.Printf("GPU MCTS failed: %v, falling back to CPU", err)
	}
	
	// CPU implementation (original)
	rootNode := &MCTSNode{
		state:        state,
		untriedMoves: simulation.GetValidMoves(state, state.You.ID),
	}
	
	startTime := time.Now()
	iterations := 0
	
	// Run MCTS iterations until time/iteration limit
	for iterations < m.MaxIterations && time.Since(startTime) < m.MaxTime {
		// 1. Selection - traverse tree using UCB
		selectedNode := m.selectNode(rootNode)
		
		// 2. Expansion - add new child if possible
		expandedNode := m.expand(selectedNode)
		
		// 3. Simulation - play out random game
		reward := m.simulate(expandedNode.state)
		
		// 4. Backpropagation - update statistics
		m.backpropagate(expandedNode, reward)
		
		iterations++
	}
	
	// Return move with highest visit count (most explored)
	return m.bestMove(rootNode)
}

// selectNode traverses tree using UCB1 formula
func (m *MCTSSearch) selectNode(node *MCTSNode) *MCTSNode {
	current := node
	
	// Traverse until we find a node with untried moves or reach leaf
	for len(current.untriedMoves) == 0 && len(current.children) > 0 {
		current = m.selectBestChild(current)
	}
	
	return current
}

// selectBestChild uses UCB1 to select child with best exploration/exploitation balance
func (m *MCTSSearch) selectBestChild(node *MCTSNode) *MCTSNode {
	bestScore := -math.MaxFloat64
	var bestChild *MCTSNode
	
	for _, child := range node.children {
		// UCB1 formula: exploitation + exploration
		exploitation := child.wins / float64(child.visits)
		exploration := m.ExplorationC * math.Sqrt(math.Log(float64(node.visits))/float64(child.visits))
		score := exploitation + exploration
		
		if score > bestScore {
			bestScore = score
			bestChild = child
		}
	}
	
	return bestChild
}

// expand adds a new child node for an untried move
func (m *MCTSSearch) expand(node *MCTSNode) *MCTSNode {
	// If no untried moves, return the node itself
	if len(node.untriedMoves) == 0 {
		return node
	}
	
	// Pick random untried move
	moveIndex := m.rng.Intn(len(node.untriedMoves))
	move := node.untriedMoves[moveIndex]
	
	// Remove from untried moves
	node.untriedMoves = append(node.untriedMoves[:moveIndex], node.untriedMoves[moveIndex+1:]...)
	
	// Simulate the move
	newState := simulation.SimulateMove(node.state, node.state.You.ID, move)
	
	// Create child node
	child := &MCTSNode{
		state:        newState,
		move:         move,
		parent:       node,
		untriedMoves: simulation.GetValidMoves(newState, newState.You.ID),
	}
	
	node.children = append(node.children, child)
	
	return child
}

// simulate plays out a random game from the given state
func (m *MCTSSearch) simulate(state *board.GameState) float64 {
	currentState := state
	depth := 0
	
	// Play random moves until game ends or max depth
	for depth < m.MaxDepth {
		// Check if our snake is still alive
		if !isSnakeAlive(currentState, currentState.You.ID) {
			return 0.0 // We lost
		}
		
		// Check if we're the only snake left
		aliveCount := 0
		for _, snake := range currentState.Board.Snakes {
			if snake.Health > 0 {
				aliveCount++
			}
		}
		if aliveCount == 1 {
			return 1.0 // We won
		}
		
		// Get valid moves
		validMoves := simulation.GetValidMoves(currentState, currentState.You.ID)
		if len(validMoves) == 0 {
			return 0.0 // No valid moves - we lose
		}
		
		// Pick random move (with slight bias toward heuristically good moves)
		move := m.selectSimulationMove(currentState, validMoves)
		currentState = simulation.SimulateMove(currentState, currentState.You.ID, move)
		
		// Simulate random enemy moves
		currentState = m.simulateEnemyMoves(currentState)
		
		depth++
	}
	
	// Game didn't end - evaluate final state heuristically
	return m.evaluateState(currentState)
}

// selectSimulationMove picks a move for simulation (with heuristic bias)
func (m *MCTSSearch) selectSimulationMove(state *board.GameState, validMoves []string) string {
	// 80% of time use heuristic best, 20% random (for exploration)
	if m.rng.Float64() < 0.8 && len(validMoves) > 1 {
		// Use greedy heuristic to guide
		bestMove := validMoves[0]
		bestScore := m.baseStrategy.ScoreMove(state, bestMove)
		
		for _, move := range validMoves[1:] {
			score := m.baseStrategy.ScoreMove(state, move)
			if score > bestScore {
				bestScore = score
				bestMove = move
			}
		}
		return bestMove
	}
	
	// Random move
	return validMoves[m.rng.Intn(len(validMoves))]
}

// simulateEnemyMoves simulates random enemy moves
func (m *MCTSSearch) simulateEnemyMoves(state *board.GameState) *board.GameState {
	newState := state
	
	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}
		
		validMoves := simulation.GetValidMoves(newState, snake.ID)
		if len(validMoves) > 0 {
			move := validMoves[m.rng.Intn(len(validMoves))]
			newState = simulation.SimulateMove(newState, snake.ID, move)
		}
	}
	
	return newState
}

// evaluateState evaluates a non-terminal game state
func (m *MCTSSearch) evaluateState(state *board.GameState) float64 {
	// Check if we're alive
	ourSnake := state.Board.GetSnakeByID(state.You.ID)
	if ourSnake == nil || ourSnake.Health <= 0 {
		return 0.0
	}
	
	// Heuristic evaluation: space, health, length
	mySpace := heuristics.EvaluateSpace(state, ourSnake.Head, 15)
	
	// Calculate enemy threat
	maxEnemyLength := 0
	for _, snake := range state.Board.Snakes {
		if snake.ID != state.You.ID && snake.Length > maxEnemyLength {
			maxEnemyLength = snake.Length
		}
	}
	
	lengthAdvantage := float64(ourSnake.Length - maxEnemyLength)
	healthScore := float64(ourSnake.Health) / 100.0
	
	// Combine factors
	score := mySpace*0.5 + healthScore*0.2 + (lengthAdvantage/10.0)*0.3
	
	// Clamp to [0, 1]
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}
	
	return score
}

// backpropagate updates node statistics up the tree
func (m *MCTSSearch) backpropagate(node *MCTSNode, reward float64) {
	current := node
	
	for current != nil {
		current.visits++
		current.wins += reward
		current = current.parent
	}
}

// bestMove returns the move with the highest visit count
func (m *MCTSSearch) bestMove(rootNode *MCTSNode) string {
	if len(rootNode.children) == 0 {
		// No children - return first valid move as fallback
		moves := simulation.GetValidMoves(rootNode.state, rootNode.state.You.ID)
		if len(moves) > 0 {
			return moves[0]
		}
		return board.MoveUp
	}
	
	bestVisits := -1
	bestMove := board.MoveUp
	
	for _, child := range rootNode.children {
		if child.visits > bestVisits {
			bestVisits = child.visits
			bestMove = child.move
		}
	}
	
	return bestMove
}

// ScoreMove provides a score (for compatibility with SearchStrategy interface)
func (m *MCTSSearch) ScoreMove(state *board.GameState, move string) float64 {
	// MCTS doesn't naturally provide per-move scores
	// Use greedy heuristic as approximation
	return m.baseStrategy.ScoreMove(state, move)
}

// isSnakeAlive checks if a snake is alive
func isSnakeAlive(state *board.GameState, snakeID string) bool {
	snake := state.Board.GetSnakeByID(snakeID)
	return snake != nil && snake.Health > 0
}
