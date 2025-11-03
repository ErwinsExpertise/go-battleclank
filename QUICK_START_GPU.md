# Quick Start: GPU Acceleration

This guide gets you running with GPU acceleration in 5 minutes.

## Prerequisites

- NVIDIA GPU (GTX 900 series or newer)
- **Linux system** (Ubuntu 20.04+, Debian, CentOS, etc.)
- Go 1.24.7 installed
- **CUDA 11.8** (CUDA 12.x is not compatible)

**Windows Users:** Use WSL2 (Windows Subsystem for Linux) for CUDA builds. See [BUILD_WITH_CUDA.md](BUILD_WITH_CUDA.md#problem-build-errors-on-windows-undefined-cufunction-cuffthandle-etc) for details.

## Step 1: Install CUDA Toolkit 11.8

**Important:** CUDA 12.x is not compatible. Use CUDA 11.8.

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA 11.8
sudo apt-get install -y cuda-toolkit-11-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify (should show release 11.8)
nvcc --version
nvidia-smi
```

## Step 2: Build with CUDA Support

```bash
cd /home/runner/work/go-battleclank/go-battleclank

# Build with CUDA
go build -tags cuda -o go-battleclank-cuda .
```

## Step 3: Run with GPU Acceleration

```bash
# Start server with GPU
./go-battleclank-cuda --enable-gpu

# You should see:
# "GPU initialized: NVIDIA GeForce RTX 3080 (device 0 of 1)"
```

## Step 4: Verify GPU is Being Used

In another terminal, monitor GPU usage:

```bash
watch -n 0.5 nvidia-smi
```

You should see GPU utilization increase when the Battlesnake makes moves.

## Step 5: Run Benchmarks

```bash
# Run benchmark with GPU
go run -tags cuda tools/live_benchmark.go -games 100 -enable-gpu -go-url http://localhost:8000
```

## Expected Performance

With GPU acceleration:
- **MCTS**: ~50ms per 100 iterations (vs ~500ms on CPU)
- **Win Rate**: +5-10% improvement
- **Decision Time**: 50-100ms (vs 300-500ms on CPU)

## Troubleshooting

### "cuda.h: No such file or directory"
CUDA Toolkit not installed. Follow Step 1 above.

### "Binary built without CUDA support"
You forgot `-tags cuda` when building. Rebuild with:
```bash
go build -tags cuda -o go-battleclank-cuda .
```

### "CUDA not available"
Check GPU drivers:
```bash
nvidia-smi  # Should show your GPU
```

### GPU not being used
Monitor GPU during gameplay:
```bash
watch -n 0.5 nvidia-smi
```
If utilization stays at 0%, check logs for errors.

## Without CUDA

If you don't have CUDA or want to test on a non-GPU system:

```bash
# Build without CUDA (works anywhere)
go build -o go-battleclank .

# Run (will use CPU fallback)
./go-battleclank --enable-gpu
```

## Complete Documentation

- [BUILD_WITH_CUDA.md](BUILD_WITH_CUDA.md) - Detailed build guide
- [GPU_USAGE.md](GPU_USAGE.md) - Complete usage documentation
- [README.md](README.md) - General project documentation

## Support

For issues or questions, see the troubleshooting sections in:
- [BUILD_WITH_CUDA.md](BUILD_WITH_CUDA.md#troubleshooting)
- [GPU_USAGE.md](GPU_USAGE.md#troubleshooting)
