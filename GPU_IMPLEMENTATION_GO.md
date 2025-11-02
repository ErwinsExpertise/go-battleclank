# GPU Implementation Guide - Option A (Go CUDA Bindings)

## Overview

This guide details implementing GPU acceleration entirely in Go using CUDA bindings, keeping the entire codebase in a single language.

## Why Go CUDA (Option A)

**Advantages**:
- ✅ No language barrier - everything stays in Go
- ✅ No IPC overhead - direct GPU calls
- ✅ Seamless integration with existing code
- ✅ Single binary deployment
- ✅ Easier debugging and maintenance
- ✅ Better performance (no serialization/RPC overhead)

**Trade-offs**:
- Requires CGO (enabled by default in Go)
- Need CUDA development environment
- Slightly more complex build process

## Available Go CUDA Libraries

### 1. mumax/3 cuda Package (Recommended)

**Repository**: https://github.com/mumax/3/tree/master/cuda

**Features**:
- Pure Go interface to CUDA
- Battle-tested in physics simulations
- Good documentation
- Active maintenance

**Installation**:
```bash
go get github.com/mumax/3/cuda
```

**Requirements**:
- CUDA Toolkit 10.0+
- NVIDIA GPU drivers
- CGO enabled (default)

### 2. barnex/cuda5 (Alternative)

**Repository**: https://github.com/barnex/cuda5

**Features**:
- Lightweight CUDA wrapper
- Simple API
- Lower-level control

### 3. InternatBlackhole/cudago (Alternative)

**Repository**: https://github.com/InternatBlackhole/cudago

**Features**:
- Modern CUDA 11+ support
- Clean API design
- Good for new projects

## Implementation Plan

### Phase 1: Setup and Infrastructure (Week 1)

#### Step 1.1: Setup CUDA Development Environment

```bash
# Install CUDA Toolkit (if not already installed)
# On Ubuntu/Debian:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# Verify installation
nvcc --version
nvidia-smi
```

#### Step 1.2: Add CUDA Dependency to Project

```bash
cd /home/runner/work/go-battleclank/go-battleclank

# Add mumax cuda package
go get github.com/mumax/3/cuda

# Update go.mod
go mod tidy
```

#### Step 1.3: Create GPU Package Structure

```
go-battleclank/
├── gpu/
│   ├── gpu.go              # GPU initialization and management
│   ├── simulation.go       # GPU-accelerated simulation kernels
│   ├── floodfill.go        # GPU flood fill implementation
│   ├── pathfinding.go      # GPU A* pathfinding
│   └── mcts.go             # GPU-accelerated MCTS
```

### Phase 2: Core GPU Primitives (Week 2)

#### 2.1: GPU Initialization Module

Create `gpu/gpu.go`:

```go
package gpu

import (
	"fmt"
	"log"
	"github.com/mumax/3/cuda"
)

var (
	// Global GPU availability flag
	GPUAvailable bool
	
	// Device info
	DeviceCount int
	DeviceName  string
)

// Initialize checks for GPU availability and initializes CUDA
func Initialize() error {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("GPU initialization failed: %v", r)
			GPUAvailable = false
		}
	}()
	
	// Check CUDA availability
	if !cuda.Available() {
		log.Println("CUDA not available, using CPU fallback")
		GPUAvailable = false
		return fmt.Errorf("CUDA not available")
	}
	
	// Get device count
	DeviceCount = cuda.DeviceCount()
	if DeviceCount == 0 {
		log.Println("No CUDA devices found, using CPU fallback")
		GPUAvailable = false
		return fmt.Errorf("no CUDA devices found")
	}
	
	// Select device 0 by default
	cuda.SetDevice(0)
	DeviceName = cuda.DeviceName(0)
	
	log.Printf("GPU initialized: %s (device 0 of %d)", DeviceName, DeviceCount)
	GPUAvailable = true
	
	return nil
}

// Cleanup releases GPU resources
func Cleanup() {
	if GPUAvailable {
		cuda.Recycle()
	}
}

// WithDevice runs a function on a specific GPU device
func WithDevice(deviceID int, fn func() error) error {
	if !GPUAvailable {
		return fmt.Errorf("GPU not available")
	}
	
	if deviceID >= DeviceCount {
		return fmt.Errorf("device %d not available (only %d devices)", deviceID, DeviceCount)
	}
	
	oldDevice := cuda.GetDevice()
	cuda.SetDevice(deviceID)
	defer cuda.SetDevice(oldDevice)
	
	return fn()
}
```

#### 2.2: Board State Representation for GPU

Create `gpu/board_state.go`:

```go
package gpu

import (
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/mumax/3/cuda"
)

// BoardStateGPU represents a board state optimized for GPU
type BoardStateGPU struct {
	Width      int
	Height     int
	Occupancy  *cuda.DevicePtr  // 2D grid: 0=empty, 1=occupied
	SnakeHeads *cuda.DevicePtr  // Snake head positions
	NumSnakes  int
}

// NewBoardStateGPU converts CPU board state to GPU format
func NewBoardStateGPU(state *board.GameState) (*BoardStateGPU, error) {
	if !GPUAvailable {
		return nil, fmt.Errorf("GPU not available")
	}
	
	w, h := state.Board.Width, state.Board.Height
	size := w * h
	
	// Create occupancy grid
	occupancy := make([]float32, size)
	for _, snake := range state.Board.Snakes {
		for _, segment := range snake.Body {
			idx := segment.Y*w + segment.X
			if idx >= 0 && idx < size {
				occupancy[idx] = 1.0
			}
		}
	}
	
	// Upload to GPU
	occupancyGPU := cuda.MemAlloc(int64(size * 4)) // 4 bytes per float32
	cuda.Memcpy(occupancyGPU, occupancy)
	
	// Create snake heads array
	numSnakes := len(state.Board.Snakes)
	heads := make([]float32, numSnakes*2) // [x0, y0, x1, y1, ...]
	for i, snake := range state.Board.Snakes {
		heads[i*2] = float32(snake.Head.X)
		heads[i*2+1] = float32(snake.Head.Y)
	}
	
	headsGPU := cuda.MemAlloc(int64(numSnakes * 2 * 4))
	cuda.Memcpy(headsGPU, heads)
	
	return &BoardStateGPU{
		Width:      w,
		Height:     h,
		Occupancy:  &occupancyGPU,
		SnakeHeads: &headsGPU,
		NumSnakes:  numSnakes,
	}, nil
}

// Free releases GPU memory
func (b *BoardStateGPU) Free() {
	if b.Occupancy != nil {
		cuda.MemFree(*b.Occupancy)
	}
	if b.SnakeHeads != nil {
		cuda.MemFree(*b.SnakeHeads)
	}
}
```

#### 2.3: GPU Flood Fill Implementation

Create `gpu/floodfill.go`:

```go
package gpu

import (
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/mumax/3/cuda"
)

// FloodFillGPU performs parallel flood fill on GPU
// Returns the number of reachable cells
func FloodFillGPU(state *board.GameState, start board.Coord, maxDepth int) (int, error) {
	if !GPUAvailable {
		return 0, fmt.Errorf("GPU not available")
	}
	
	// Convert board to GPU format
	boardGPU, err := NewBoardStateGPU(state)
	if err != nil {
		return 0, err
	}
	defer boardGPU.Free()
	
	// Allocate visited and frontier arrays
	size := boardGPU.Width * boardGPU.Height
	visited := cuda.MemAlloc(int64(size * 4))
	defer cuda.MemFree(visited)
	
	frontier := cuda.MemAlloc(int64(size * 4))
	defer cuda.MemFree(frontier)
	
	// Initialize: mark start position
	startIdx := start.Y*boardGPU.Width + start.X
	initialFrontier := make([]float32, size)
	initialFrontier[startIdx] = 1.0
	cuda.Memcpy(frontier, initialFrontier)
	
	// Run BFS iterations on GPU
	depth := 0
	for depth < maxDepth {
		// Launch kernel to expand frontier
		// This would use a custom CUDA kernel
		// For now, we'll use a simplified version
		
		// TODO: Implement actual CUDA kernel for parallel BFS
		// kernel<<<blocks, threads>>>(occupancy, visited, frontier, width, height)
		
		depth++
	}
	
	// Count visited cells
	visitedCells := make([]float32, size)
	cuda.MemcpyDtoH(visitedCells, visited)
	
	count := 0
	for _, v := range visitedCells {
		if v > 0 {
			count++
		}
	}
	
	return count, nil
}

// Note: This is a simplified version. A production implementation
// would include custom CUDA kernels for optimal performance.
// The mumax/3 package provides ways to load and run custom kernels.
```

### Phase 3: MCTS GPU Acceleration (Week 3-4)

#### 3.1: Batch MCTS Implementation

Create `gpu/mcts.go`:

```go
package gpu

import (
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/mumax/3/cuda"
	"math"
	"sync"
)

// MCTSBatchGPU runs multiple MCTS simulations in parallel on GPU
type MCTSBatchGPU struct {
	batchSize     int
	maxDepth      int
	explorationC  float64
	
	// GPU memory pools
	statePool     []*cuda.DevicePtr
	resultPool    []*cuda.DevicePtr
}

// NewMCTSBatchGPU creates a new batch MCTS runner
func NewMCTSBatchGPU(batchSize, maxDepth int) *MCTSBatchGPU {
	return &MCTSBatchGPU{
		batchSize:    batchSize,
		maxDepth:     maxDepth,
		explorationC: math.Sqrt(2),
	}
}

// SimulateBatch runs multiple MCTS simulations in parallel
func (m *MCTSBatchGPU) SimulateBatch(state *board.GameState, iterations int) (string, error) {
	if !GPUAvailable {
		return "", fmt.Errorf("GPU not available")
	}
	
	// Prepare batch of simulations
	numBatches := (iterations + m.batchSize - 1) / m.batchSize
	
	results := make([]MCTSResult, 4) // 4 possible moves
	var wg sync.WaitGroup
	
	for batch := 0; batch < numBatches; batch++ {
		currentBatchSize := m.batchSize
		if batch == numBatches-1 {
			currentBatchSize = iterations - batch*m.batchSize
		}
		
		// Launch batch on GPU
		wg.Add(1)
		go func(batchIdx int) {
			defer wg.Done()
			m.runBatchOnGPU(state, currentBatchSize, batchIdx, results)
		}(batch)
	}
	
	wg.Wait()
	
	// Select best move based on aggregated results
	bestMove := selectBestMove(results)
	return bestMove, nil
}

// runBatchOnGPU executes a batch of simulations on GPU
func (m *MCTSBatchGPU) runBatchOnGPU(state *board.GameState, batchSize, batchIdx int, results []MCTSResult) {
	// Convert state to GPU format
	boardGPU, err := NewBoardStateGPU(state)
	if err != nil {
		return
	}
	defer boardGPU.Free()
	
	// Allocate result buffer
	resultSize := batchSize * 4 * 2 // 4 moves × (visits, wins)
	resultBuffer := cuda.MemAlloc(int64(resultSize * 4))
	defer cuda.MemFree(resultBuffer)
	
	// Launch CUDA kernel for parallel simulation
	// TODO: Implement custom CUDA kernel
	// mctsKernel<<<blocks, threads>>>(boardGPU, resultBuffer, batchSize, maxDepth)
	
	// Copy results back
	resultsHost := make([]float32, resultSize)
	cuda.MemcpyDtoH(resultsHost, resultBuffer)
	
	// Aggregate results
	// ... (implementation details)
}

type MCTSResult struct {
	Move   string
	Visits int
	Wins   float64
}

func selectBestMove(results []MCTSResult) string {
	bestIdx := 0
	bestVisits := results[0].Visits
	
	for i := 1; i < len(results); i++ {
		if results[i].Visits > bestVisits {
			bestIdx = i
			bestVisits = results[i].Visits
		}
	}
	
	moves := []string{"up", "down", "left", "right"}
	return moves[bestIdx]
}
```

### Phase 4: Integration with Existing Code (Week 5)

#### 4.1: Update MCTS Search Algorithm

Modify `algorithms/search/mcts.go`:

```go
package search

import (
	"github.com/ErwinsExpertise/go-battleclank/engine/board"
	"github.com/ErwinsExpertise/go-battleclank/gpu"
	"time"
)

// FindBestMove uses MCTS to find the best move
func (m *MCTSSearch) FindBestMove(state *board.GameState) string {
	// Try GPU acceleration first
	if gpu.GPUAvailable {
		gpuMCTS := gpu.NewMCTSBatchGPU(100, m.MaxDepth)
		move, err := gpuMCTS.SimulateBatch(state, m.MaxIterations)
		if err == nil {
			return move
		}
		// Fallback to CPU if GPU fails
		log.Printf("GPU MCTS failed: %v, falling back to CPU", err)
	}
	
	// Original CPU implementation as fallback
	rootNode := &MCTSNode{
		state:        state,
		untriedMoves: simulation.GetValidMoves(state, state.You.ID),
	}
	
	// ... (rest of CPU implementation)
}
```

#### 4.2: Update Main Initialization

Modify `main.go`:

```go
package main

import (
	"log"
	"github.com/ErwinsExpertise/go-battleclank/gpu"
)

func main() {
	// Initialize GPU if available
	if err := gpu.Initialize(); err != nil {
		log.Printf("GPU initialization failed: %v (will use CPU)", err)
	} else {
		log.Printf("GPU acceleration enabled: %s", gpu.DeviceName)
		defer gpu.Cleanup()
	}
	
	// Rest of initialization...
	RunServer()
}
```

### Phase 5: Custom CUDA Kernels (Week 6)

For maximum performance, implement custom CUDA kernels:

#### 5.1: Flood Fill Kernel

Create `gpu/kernels/floodfill.cu`:

```cuda
__global__ void floodFillKernel(
    const float* occupancy,
    float* visited,
    float* frontier,
    float* newFrontier,
    int width,
    int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    
    if (idx >= total) return;
    
    // If this cell is in frontier
    if (frontier[idx] > 0 && visited[idx] == 0) {
        visited[idx] = 1.0;
        
        // Mark neighbors as new frontier
        int x = idx % width;
        int y = idx / width;
        
        // Up
        if (y + 1 < height) {
            int upIdx = (y + 1) * width + x;
            if (occupancy[upIdx] == 0) {
                newFrontier[upIdx] = 1.0;
            }
        }
        
        // Down
        if (y > 0) {
            int downIdx = (y - 1) * width + x;
            if (occupancy[downIdx] == 0) {
                newFrontier[downIdx] = 1.0;
            }
        }
        
        // Left
        if (x > 0) {
            int leftIdx = y * width + (x - 1);
            if (occupancy[leftIdx] == 0) {
                newFrontier[leftIdx] = 1.0;
            }
        }
        
        // Right
        if (x + 1 < width) {
            int rightIdx = y * width + (x + 1);
            if (occupancy[rightIdx] == 0) {
                newFrontier[rightIdx] = 1.0;
            }
        }
    }
}
```

#### 5.2: Load Custom Kernel in Go

```go
package gpu

import (
	"github.com/mumax/3/cuda"
)

var floodFillKernel cuda.Function

func init() {
	if GPUAvailable {
		// Load custom kernel from PTX or CUBIN
		module := cuda.ModuleLoad("kernels/floodfill.ptx")
		floodFillKernel = module.GetFunction("floodFillKernel")
	}
}

func runFloodFillKernel(occupancy, visited, frontier, newFrontier cuda.DevicePtr, width, height int) {
	threads := 256
	blocks := (width*height + threads - 1) / threads
	
	cfg := cuda.LaunchConfig{
		GridDim:  cuda.Dim3{X: blocks},
		BlockDim: cuda.Dim3{X: threads},
	}
	
	cuda.LaunchKernel(floodFillKernel, cfg,
		occupancy, visited, frontier, newFrontier, width, height)
}
```

## Build Configuration

### Update Build Scripts

Create `build-gpu.sh`:

```bash
#!/bin/bash

# Build with GPU support
echo "Building with GPU support..."

# Ensure CUDA is in path
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Enable CGO for CUDA bindings
export CGO_ENABLED=1

# Build
go build -tags gpu -o battlesnake-gpu .

echo "Build complete: battlesnake-gpu"
```

### Conditional Compilation

Use build tags for GPU features:

```go
//go:build gpu
// +build gpu

package gpu

// GPU-specific code here
```

```go
//go:build !gpu
// +build !gpu

package gpu

// CPU fallback stubs
func Initialize() error {
	return fmt.Errorf("built without GPU support")
}
```

## Testing

### Unit Tests

Create `gpu/gpu_test.go`:

```go
package gpu

import (
	"testing"
)

func TestGPUInitialization(t *testing.T) {
	err := Initialize()
	if err != nil {
		t.Skip("GPU not available, skipping test")
	}
	
	if !GPUAvailable {
		t.Fatal("GPU should be available after successful init")
	}
	
	Cleanup()
}

func BenchmarkFloodFillGPU(b *testing.B) {
	// Benchmark GPU vs CPU flood fill
	// ...
}
```

### Integration Tests

```bash
# Run GPU tests only on machines with GPU
go test -tags gpu ./gpu/... -v

# Run all tests with CPU fallback
go test ./... -v
```

## Deployment

### Docker with GPU Support

Create `Dockerfile.gpu`:

```dockerfile
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Install Go
RUN apt-get update && apt-get install -y wget
RUN wget https://go.dev/dl/go1.24.7.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.24.7.linux-amd64.tar.gz
ENV PATH=$PATH:/usr/local/go/bin

WORKDIR /app
COPY . .

# Build with GPU support
RUN CGO_ENABLED=1 go build -tags gpu -o battlesnake-gpu .

EXPOSE 8000
CMD ["./battlesnake-gpu"]
```

### Run with GPU

```bash
# Build
docker build -f Dockerfile.gpu -t battlesnake-gpu .

# Run with GPU access
docker run --gpus all -p 8000:8000 battlesnake-gpu
```

## Performance Expectations

### MCTS Speedup

```
CPU (sequential):  100 iterations × 15 depth = ~500ms
GPU (parallel):    100 iterations × 15 depth = ~50ms
Speedup:          10x
```

### Flood Fill Speedup

```
CPU: ~1-2ms per call
GPU: ~0.1-0.2ms per call (amortized over batch)
Speedup: 5-10x
```

### Overall Game Performance

```
Move decision time:
- CPU only: 300-500ms (frequent timeouts)
- GPU accelerated: 50-100ms (reliable, complete search)

Win rate impact:
- Better decisions from complete MCTS
- Estimated: +5-10 percentage points
```

## Fallback Strategy

The implementation maintains CPU fallback at every level:

```go
func SomeGPUFunction(args) result {
	if gpu.GPUAvailable {
		result, err := gpuImplementation(args)
		if err == nil {
			return result
		}
		log.Printf("GPU failed: %v, falling back to CPU", err)
	}
	
	// CPU implementation
	return cpuImplementation(args)
}
```

This ensures:
- ✅ Works on machines without GPU
- ✅ Graceful degradation on GPU errors
- ✅ Easy testing on development machines
- ✅ Production reliability

## Summary

**Option A (Go CUDA Bindings)** provides:

✅ **Pure Go solution** - No language mixing  
✅ **Direct GPU access** - No IPC overhead  
✅ **Seamless integration** - Minimal code changes  
✅ **Better performance** - Lower latency  
✅ **Single binary** - Easy deployment  
✅ **Robust fallback** - CPU implementation always available  

**Estimated Timeline**: 6 weeks to full implementation  
**Estimated Performance Gain**: 5-10x for MCTS, +5-10% win rate  
**Complexity**: Medium (requires CUDA experience)  
**Maintainability**: High (single language, clear architecture)
