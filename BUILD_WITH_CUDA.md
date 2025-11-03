# Building with CUDA Support

This document describes how to build go-battleclank with GPU acceleration using CUDA.

## Platform Support

**Supported Platforms:**
- ✅ Linux (Ubuntu, Debian, CentOS, etc.)
- ⚠️ Windows - **Known Issues** (see Windows section below)
- ⚠️ macOS - Limited support (NVIDIA drivers required)

**Note:** The mumax/3 CUDA bindings used in this project are primarily developed and tested on Linux. Windows users may encounter build errors. See the Windows section below for workarounds.

## Prerequisites

### 1. CUDA Toolkit Installation

You need CUDA Toolkit 10.0 or higher installed on your system.

#### Ubuntu/Debian:
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install cuda-toolkit-12-3

# Verify installation
nvcc --version
nvidia-smi
```

#### Other Linux Distributions:
Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) and follow the instructions for your distribution.

### 2. Set Environment Variables

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CGO_ENABLED=1
```

Reload your shell:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### 3. Verify CUDA Installation

```bash
# Check CUDA compiler
nvcc --version

# Check GPU is detected
nvidia-smi

# Verify CGO is enabled
go env CGO_ENABLED  # Should output: 1
```

## Building with CUDA Support

### Standard Build with CUDA

Use the `cuda` build tag to enable GPU acceleration:

```bash
cd /home/runner/work/go-battleclank/go-battleclank
go build -tags cuda -o go-battleclank-cuda .
```

### Build Options

**Release build with optimizations:**
```bash
go build -tags cuda -ldflags="-s -w" -o go-battleclank-cuda .
```

**Debug build with symbols:**
```bash
go build -tags cuda -gcflags="all=-N -l" -o go-battleclank-cuda-debug .
```

**Static binary (if possible):**
```bash
CGO_ENABLED=1 go build -tags cuda -ldflags="-linkmode external -extldflags -static" -o go-battleclank-cuda .
```
Note: Static linking may not work due to CUDA dependencies.

### Verify CUDA Build

Check that the binary is using CUDA:

```bash
# Run with GPU enabled
./go-battleclank-cuda --enable-gpu

# Should see:
# "GPU initialized: NVIDIA GeForce RTX 3080 (device 0 of 1)"
# NOT: "Binary built without CUDA support"
```

## Building Without CUDA (Fallback)

To build a binary that works on systems without CUDA:

```bash
go build -o go-battleclank .
```

This will build with CPU fallback only. The binary will work on any system but won't use GPU acceleration.

## Running with GPU Acceleration

Once built with CUDA support:

```bash
# Start server with GPU acceleration
./go-battleclank-cuda --enable-gpu

# Run benchmark with GPU
go run -tags cuda tools/live_benchmark.go -games 100 -enable-gpu
```

## Testing CUDA Build

Run tests with CUDA enabled:

```bash
# Unit tests
go test -tags cuda ./gpu/... -v

# All tests
go test -tags cuda ./... -v

# With race detector
go test -tags cuda -race ./gpu/... -v
```

## Docker Build with CUDA

Create a Dockerfile for CUDA support:

```dockerfile
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Install Go
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://go.dev/dl/go1.24.7.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.24.7.linux-amd64.tar.gz && \
    rm go1.24.7.linux-amd64.tar.gz

ENV PATH=$PATH:/usr/local/go/bin
ENV CGO_ENABLED=1

WORKDIR /app
COPY . .

# Build with CUDA support
RUN go build -tags cuda -o go-battleclank-cuda .

EXPOSE 8000
CMD ["./go-battleclank-cuda", "--enable-gpu"]
```

Build and run:
```bash
docker build -t go-battleclank-cuda .
docker run --gpus all -p 8000:8000 go-battleclank-cuda
```

## Troubleshooting

### Problem: "cuda.h: No such file or directory"

**Solution:** CUDA Toolkit is not installed or not in the system path.
```bash
# Verify CUDA installation
ls /usr/local/cuda/include/cuda.h

# If missing, install CUDA Toolkit (see Prerequisites section)
```

### Problem: "cannot find -lcuda"

**Solution:** CUDA libraries are not in the library path.
```bash
# Add to ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

### Problem: "Binary built without CUDA support"

**Solution:** You forgot the `-tags cuda` flag when building.
```bash
# Rebuild with CUDA tag
go build -tags cuda -o go-battleclank-cuda .
```

### Problem: GPU initialization fails at runtime

**Solution:** Check GPU drivers and CUDA runtime.
```bash
# Verify GPU is accessible
nvidia-smi

# Check CUDA runtime version
cat /usr/local/cuda/version.txt

# Verify drivers match CUDA version
nvidia-smi | grep "CUDA Version"
```

### Problem: "GPU initialized" but performance is same as CPU

**Solution:** Verify GPU is actually being used:
```bash
# In another terminal, monitor GPU usage while running
watch -n 0.5 nvidia-smi

# You should see GPU utilization increase during MCTS operations
```

### Problem: Build errors on Windows (undefined: cu.Function, cufft.Handle, etc.)

**Issue:** The mumax/3 CUDA bindings have known compatibility issues on Windows due to CGO and CUDA header path differences.

**Solutions:**

1. **Use WSL2 (Recommended):**
   ```powershell
   # Install WSL2 with Ubuntu
   wsl --install -d Ubuntu
   
   # Inside WSL2, install CUDA Toolkit
   # Follow the Linux instructions above
   ```

2. **Use Linux VM or Container:**
   - Run the CUDA build in a Linux container or VM
   - Docker with NVIDIA Container Toolkit (see Docker section)

3. **Alternative: Use CPU build on Windows:**
   ```bash
   # Build without CUDA tags (works on Windows)
   go build -o go-battleclank.exe .
   ./go-battleclank.exe --enable-gpu  # Will use CPU fallback
   ```

4. **Future Alternative:**
   - Consider using a different CUDA binding library with better Windows support
   - Track issue: https://github.com/mumax/3/issues for Windows support

**Current Recommendation for Windows Users:**
Use WSL2 with Linux for CUDA builds, or use the CPU-only build. The GPU acceleration features are designed for Linux environments where CUDA integration is more stable.

## Performance Verification

To verify GPU acceleration is working:

1. **Build both versions:**
   ```bash
   go build -o go-battleclank-cpu .
   go build -tags cuda -o go-battleclank-cuda .
   ```

2. **Run benchmark comparison:**
   ```bash
   # CPU version
   ./go-battleclank-cpu --enable-gpu  # Will use CPU fallback
   
   # CUDA version
   ./go-battleclank-cuda --enable-gpu  # Will use GPU
   ```

3. **Monitor GPU usage:**
   ```bash
   watch -n 0.5 nvidia-smi
   ```

4. **Expected results:**
   - GPU version should show 5-10x faster MCTS
   - nvidia-smi should show GPU utilization during moves
   - Win rate should improve by ~5-10%

## Build Scripts

Create `build-cuda.sh` for convenience:

```bash
#!/bin/bash
set -e

echo "Building go-battleclank with CUDA support..."

# Verify CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Install CUDA Toolkit first."
    exit 1
fi

# Build with CUDA
export CGO_ENABLED=1
go build -tags cuda -v -o go-battleclank-cuda .

echo "Build complete: go-battleclank-cuda"
echo "Run with: ./go-battleclank-cuda --enable-gpu"
```

Make it executable:
```bash
chmod +x build-cuda.sh
./build-cuda.sh
```

## Continuous Integration

For CI/CD with CUDA support, use GitHub Actions with GPU runners or configure your CI to use NVIDIA Docker:

```yaml
name: Build with CUDA
on: [push]

jobs:
  build-cuda:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.3.0-devel-ubuntu22.04
    steps:
      - uses: actions/checkout@v3
      - name: Install Go
        run: |
          wget https://go.dev/dl/go1.24.7.linux-amd64.tar.gz
          tar -C /usr/local -xzf go1.24.7.linux-amd64.tar.gz
      - name: Build
        run: |
          export PATH=$PATH:/usr/local/go/bin
          export CGO_ENABLED=1
          go build -tags cuda -v .
```

## Additional Resources

- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [mumax/3 CUDA Bindings](https://github.com/mumax/3)
- [Go CGO Documentation](https://pkg.go.dev/cmd/cgo)
- [NVIDIA GPU Support](https://www.nvidia.com/en-us/drivers/)

## License

Same as main project (MIT)
