# GPU Acceleration Plan for MCTS and Decision Modules

## Current Status

The battlesnake currently runs on CPU, with GPU resources (8× NVIDIA A100 GPUs) underutilized during training. MCTS previously failed due to CPU bottlenecks.

## Problem Analysis

### CPU Bottlenecks
- **MCTS iterations**: Heavy simulation load (100+ iterations per move)
- **Pathfinding (A*)**: Node expansion can explore 200-500 nodes
- **Flood fill**: Recursive space evaluation up to depth 100
- **Multiple snake simulations**: Predicting opponent moves

### Current GPU Usage
- **Training phase**: ~12% single GPU utilization (LLM inference only)
- **Parallel configs**: Up to 100% when using `--parallel-configs 8`
- **Inference**: Neural network pattern recognition (minimal)

## Acceleration Strategies

### 1. Batch MCTS Simulations (High Priority)

**Current**: Sequential MCTS simulations on CPU
**Proposed**: Parallel batch simulation on GPU

```
CPU: 100 iterations × 15 depth = 1500 sequential simulations
GPU: 100 iterations batched = 100 parallel simulations × 15 depth
Speedup: ~10-15x for MCTS decision time
```

**Implementation Options**:

#### Option A: Go CUDA Bindings (Recommended)
- Use `github.com/mumax/3/cuda` or similar Go CUDA library
- Keep existing Go codebase
- Add GPU kernels for simulation primitives
- Complexity: Medium, Integration: Seamless

#### Option B: Shared Python Inference Service
- Create Python microservice for GPU-accelerated MCTS
- Go battlesnake calls Python service via HTTP/gRPC
- Python uses PyTorch/CuPy for GPU acceleration
- Complexity: Low, Integration: Requires IPC overhead

#### Option C: Hybrid Approach
- Keep greedy/lookahead in Go (fast enough on CPU)
- Move MCTS to GPU-accelerated Python service
- Use hybrid algorithm to decide when to call GPU service
- Complexity: Low, Integration: Moderate IPC overhead

### 2. Vectorized Board State Operations

**Current**: Board state manipulations in Go structs
**Proposed**: Batch board operations on GPU

Operations to vectorize:
- Collision detection (check if position occupied)
- Valid move generation (4 directions × N snakes)
- Flood fill (breadth-first search)
- Distance calculations (Manhattan distance to food/enemies)

**GPU Kernel Design**:
```python
# Pseudocode for GPU flood fill
def gpu_flood_fill_batch(board_states, start_positions, max_depth):
    """
    board_states: [batch_size, height, width] - occupancy grid
    start_positions: [batch_size, 2] - starting coordinates
    Returns: [batch_size] - reachable space count
    """
    # Parallel BFS using GPU threads
    # Each thread handles one board state
    # Returns space counts in parallel
```

### 3. Neural Network Integration

**Current**: Pattern recognition network on GPU (underutilized)
**Proposed**: Real-time move evaluation network

Train a policy network to guide MCTS:
- Input: Board state (11×11 grid + metadata)
- Output: Move probabilities [up, down, left, right]
- Use network to prioritize MCTS exploration

**Benefits**:
- Smarter MCTS tree exploration
- Fewer simulations needed for same quality
- GPU constantly utilized during gameplay

### 4. Multi-GPU Training Parallelism

**Current**: Sequential config testing or up to 8 parallel configs
**Proposed**: Enhanced parallel training strategies

#### Strategy A: Algorithm Diversity Testing
```bash
# Test multiple algorithms simultaneously
GPU 0-1: MCTS with config A
GPU 2-3: Hybrid with config B
GPU 4-5: Lookahead with config C
GPU 6-7: Greedy with config D
```

#### Strategy B: Ensemble Evaluation
```bash
# Create ensemble of strategies
Each GPU runs different strategy
Combine results via voting or averaging
More robust decision-making
```

## Implementation Roadmap

### Phase 1: Infrastructure (Week 1-2)
- [ ] Set up Python GPU inference service
- [ ] Implement gRPC/HTTP communication layer
- [ ] Create board state serialization format
- [ ] Benchmark communication overhead

### Phase 2: MCTS GPU Acceleration (Week 3-4)
- [ ] Implement batch simulation on GPU
- [ ] Port key primitives (collision, flood fill) to GPU
- [ ] Integrate with existing MCTS code
- [ ] Performance benchmarking

### Phase 3: Integration & Testing (Week 5-6)
- [ ] Add GPU fallback (use CPU if GPU unavailable)
- [ ] Test in continuous training loop
- [ ] Measure win rate improvements
- [ ] Optimize batch sizes for A100

### Phase 4: Advanced Features (Week 7-8)
- [ ] Train policy network for move guidance
- [ ] Implement multi-GPU ensemble strategies
- [ ] Add real-time GPU monitoring
- [ ] Documentation and deployment guide

## Expected Performance Gains

### MCTS Acceleration
```
Current: ~500ms per move (MCTS timeout, often incomplete)
GPU:     ~50-100ms per move (complete MCTS tree)
Speedup: 5-10x
```

### Training Speed
```
Current: ~30 sec/game = ~15 min/iteration (30 games)
GPU:     ~10 sec/game = ~5 min/iteration (30 games)
Speedup: 3x for iteration time
```

### Win Rate Impact
```
More complete MCTS search → Better move quality
Estimated improvement: +5-10% win rate
Baseline: 47% → Target: 52-57%
```

## Technical Specifications

### GPU Requirements
- **Current**: 8× NVIDIA A100 (40GB or 80GB)
- **Memory per simulation**: ~1-2 MB
- **Batch size**: 1000-10000 simulations (fits in GPU memory)
- **Inference time**: <1ms per batch

### Communication Protocol
```protobuf
// Example gRPC service definition
service BattlesnakeMCTS {
  rpc FindBestMove(GameStateRequest) returns (MoveResponse);
  rpc BatchSimulate(BatchSimulationRequest) returns (SimulationResults);
}

message GameStateRequest {
  int32 width = 1;
  int32 height = 2;
  repeated Snake snakes = 3;
  repeated Coord food = 4;
  int32 turn = 5;
}
```

### Fallback Strategy
```go
// Pseudocode for GPU acceleration with fallback
func FindBestMove(state *GameState) string {
    if gpuService.Available() {
        result, err := gpuService.MCTSSearch(state)
        if err == nil {
            return result.BestMove
        }
        log.Warn("GPU service failed, falling back to CPU")
    }
    // Fallback to CPU implementation
    return cpuMCTS.FindBestMove(state)
}
```

## Risks and Mitigations

### Risk 1: Communication Overhead
**Problem**: IPC latency negates GPU speedup
**Mitigation**: 
- Batch multiple requests
- Use shared memory if possible
- Profile and optimize serialization

### Risk 2: GPU Service Reliability
**Problem**: Service crashes break the snake
**Mitigation**:
- Implement robust fallback to CPU
- Health checks and auto-restart
- Timeout-based failover

### Risk 3: Increased Complexity
**Problem**: Harder to debug and maintain
**Mitigation**:
- Keep CPU implementation as reference
- Comprehensive testing
- Good documentation

## Monitoring and Metrics

### Performance Metrics
- MCTS iterations completed per move
- GPU utilization percentage
- Average move decision time
- Win rate vs baseline

### GPU Monitoring
```bash
# Track GPU usage during training
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1

# Expected: 80-100% utilization with GPU acceleration
```

### Success Criteria
- ✅ MCTS completes 100+ iterations within 300ms timeout
- ✅ GPU utilization >70% during gameplay
- ✅ Win rate improves by 5+ percentage points
- ✅ No increase in game timeout/crashes

## Cost-Benefit Analysis

### Development Cost
- Engineering time: 6-8 weeks
- Testing and validation: 2 weeks
- Total effort: ~2 person-months

### Compute Cost
- A100 GPU time: Already available (8 GPUs)
- Additional cost: None (using existing resources)

### Expected Benefit
- Win rate improvement: +5-10 percentage points
- Training speed: 3x faster (50% time saved)
- MCTS reliability: From 50% timeouts to <5% timeouts

### ROI
- Faster training convergence: Save ~60 hours per 1000 iterations
- Better decisions: More competitive in tournaments
- Scalability: Can handle larger boards and more complex scenarios

## Alternative Approaches

### 1. CPU Optimization (Lower effort, lower gain)
- Profile and optimize CPU MCTS
- Use concurrent goroutines
- Reduce simulation depth
- Expected gain: 2x speedup (still CPU-bound)

### 2. Approximate MCTS (Medium effort, medium gain)
- Use learned value functions instead of full simulation
- Lighter weight but less accurate
- Expected gain: 5x speedup, but lower quality

### 3. Full GPU Port (High effort, high gain)
- Rewrite entire snake in CUDA/Python
- Maximum performance but requires complete rewrite
- Not recommended: Too much effort for incremental gain

## Conclusion

GPU acceleration is feasible and beneficial, especially for MCTS. The recommended approach is:

1. **Start with Python inference service** (Option B) - Lower risk, faster implementation
2. **Focus on batch MCTS simulations** - Highest impact
3. **Implement robust fallback** - Maintain reliability
4. **Measure and iterate** - Data-driven optimization

Expected timeline: 6-8 weeks for full implementation
Expected outcome: 5-10% win rate improvement, 3x faster training
