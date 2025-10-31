package heuristics

import (
	"container/heap"
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"math"
)

// Package heuristics - food seeking and evaluation

// FindNearestFoodManhattan finds the nearest food using Manhattan distance
func FindNearestFoodManhattan(state *board.GameState, from board.Coord) (board.Coord, int) {
	if len(state.Board.Food) == 0 {
		return board.Coord{}, -1
	}
	
	minDist := math.MaxInt32
	var nearestFood board.Coord
	
	for _, food := range state.Board.Food {
		dist := board.ManhattanDistance(from, food)
		if dist < minDist {
			minDist = dist
			nearestFood = food
		}
	}
	
	return nearestFood, minDist
}

// IsFoodDangerous checks if food is too close to enemy snakes
func IsFoodDangerous(state *board.GameState, food board.Coord, dangerRadius int) bool {
	myDist := board.ManhattanDistance(state.You.Head, food)
	isCritical := state.You.Health < 30
	
	for _, snake := range state.Board.Snakes {
		if snake.ID == state.You.ID {
			continue
		}
		
		// Check distance to enemy body segments
		for _, segment := range snake.Body {
			dist := board.ManhattanDistance(food, segment)
			if dist <= dangerRadius {
				return true
			}
		}
		
		// Check if enemy can reach significantly faster
		enemyDist := board.ManhattanDistance(snake.Head, food)
		if enemyDist < myDist-2 {
			return true
		}
		
		// Check if arriving at same time with size disadvantage
		lengthAdvantage := snake.Length - state.You.Length
		if abs(enemyDist-myDist) <= 1 {
			if isCritical {
				if lengthAdvantage >= 3 {
					return true
				}
			} else {
				if lengthAdvantage >= 2 {
					return true
				}
			}
		}
	}
	
	return false
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// EvaluateFoodProximity scores based on distance to nearest safe food
func EvaluateFoodProximity(state *board.GameState, pos board.Coord, useAStar bool, maxAStarNodes int) float64 {
	if len(state.Board.Food) == 0 {
		return 0
	}
	
	var nearestFood board.Coord
	var distance int
	
	if useAStar && state.You.Health < 50 {
		// Use A* for more accurate pathfinding when health is low
		food, path := FindNearestFoodAStar(state, pos, maxAStarNodes)
		if path != nil && len(path) > 0 {
			nearestFood = food
			distance = len(path)
		} else {
			// Fallback to Manhattan
			nearestFood, distance = FindNearestFoodManhattan(state, pos)
		}
	} else {
		nearestFood, distance = FindNearestFoodManhattan(state, pos)
	}
	
	// Check if food is dangerous
	if IsFoodDangerous(state, nearestFood, 2) {
		// Reduce score for dangerous food
		if distance == 0 {
			return 0.1
		}
		return (1.0 / float64(distance)) * 0.1
	}
	
	// Normal food score
	if distance == 0 {
		return 1.0
	}
	return 1.0 / float64(distance)
}

// A* implementation for pathfinding

type aStarNode struct {
	pos    board.Coord
	gScore int
	fScore int
	parent *aStarNode
	index  int
}

type priorityQueue []*aStarNode

func (pq priorityQueue) Len() int           { return len(pq) }
func (pq priorityQueue) Less(i, j int) bool { return pq[i].fScore < pq[j].fScore }
func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueue) Push(x interface{}) {
	n := len(*pq)
	node := x.(*aStarNode)
	node.index = n
	*pq = append(*pq, node)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	node := old[n-1]
	old[n-1] = nil
	node.index = -1
	*pq = old[0 : n-1]
	return node
}

// FindNearestFoodAStar finds nearest food using A* pathfinding
func FindNearestFoodAStar(state *board.GameState, start board.Coord, maxNodes int) (board.Coord, []board.Coord) {
	var nearestFood board.Coord
	var shortestPath []board.Coord
	shortestLength := math.MaxInt32
	
	// Try A* to each food
	for _, food := range state.Board.Food {
		path := aStarSearch(state, start, food, maxNodes)
		if path != nil && len(path) < shortestLength {
			shortestPath = path
			nearestFood = food
			shortestLength = len(path)
		}
	}
	
	return nearestFood, shortestPath
}

func aStarSearch(state *board.GameState, start, goal board.Coord, maxNodes int) []board.Coord {
	// Check if goal is blocked
	if state.Board.IsOccupied(goal, true) {
		return nil
	}
	
	if start.X == goal.X && start.Y == goal.Y {
		return []board.Coord{start}
	}
	
	openSet := &priorityQueue{}
	heap.Init(openSet)
	
	startNode := &aStarNode{
		pos:    start,
		gScore: 0,
		fScore: board.ManhattanDistance(start, goal),
		parent: nil,
	}
	heap.Push(openSet, startNode)
	
	visited := make(map[board.Coord]int)
	nodesExplored := 0
	
	for openSet.Len() > 0 && nodesExplored < maxNodes {
		current := heap.Pop(openSet).(*aStarNode)
		nodesExplored++
		
		if current.pos.X == goal.X && current.pos.Y == goal.Y {
			return reconstructPath(current)
		}
		
		visited[current.pos] = current.gScore
		
		for _, neighbor := range state.Board.GetNeighbors(current.pos) {
			if state.Board.IsOccupied(neighbor, true) {
				continue
			}
			
			tentativeGScore := current.gScore + 1
			
			if prevScore, seen := visited[neighbor]; seen && prevScore <= tentativeGScore {
				continue
			}
			
			neighborNode := &aStarNode{
				pos:    neighbor,
				gScore: tentativeGScore,
				fScore: tentativeGScore + board.ManhattanDistance(neighbor, goal),
				parent: current,
			}
			heap.Push(openSet, neighborNode)
		}
	}
	
	return nil
}

func reconstructPath(node *aStarNode) []board.Coord {
	path := []board.Coord{}
	current := node
	
	for current != nil {
		path = append(path, current.pos)
		current = current.parent
	}
	
	// Reverse path
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	
	return path
}
