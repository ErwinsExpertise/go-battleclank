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

This project supports both CUDA 11.8 and CUDA 12.x:

- **CUDA 11.8**: Works with mumax/3 v3.9.3 (default, easiest setup)
- **CUDA 12.x**: Requires mumax/3 v3.11.1 (manual configuration needed)

#### Option A: CUDA 11.8 (Recommended - Works Out of the Box)

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA Toolkit 11.8
sudo apt-get install cuda-toolkit-11-8

# Set environment variables
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Verify installation
nvcc --version  # Should show "release 11.8"
nvidia-smi

# Build immediately works
go build -tags cuda -o go-battleclank-cuda .
```

#### Option B: CUDA 12.x with mumax/3 v3.11.1

If you have CUDA 12.x and prefer not to downgrade, you can use mumax/3 v3.11.1 which supports CUDA 12.9+.

**Note:** Due to Go module versioning issues with mumax/3 v3.11.1, manual configuration is required.

```bash
# 1. Clone mumax/3 to find the v3.11.1 commit hash
git clone --depth=50 https://github.com/mumax/3.git /tmp/mumax3
cd /tmp/mumax3
git log --all --oneline | grep -i "3.11" | head -5

# 2. Add replace directive to go.mod using the commit hash
# Edit go.mod and add at the bottom:
#   replace github.com/mumax/3 => github.com/mumax/3 <commit-hash-from-step-1>

# 3. Update dependencies and build
go mod tidy
go build -tags cuda -o go-battleclank-cuda .
```

**See the "CUDA 12.x with mumax/3 v3.11.1" section below for detailed instructions.**

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

### Problem: Build errors with CUDA 12.x (cuCtxCreate API mismatch)

**Issue:** You see errors like:
```
not enough arguments in call to (_Cfunc_cuCtxCreate)
    have (*_Ctype_CUcontext, _Ctype_uint, _Ctype_CUdevice)
    want (*_Ctype_CUcontext, *_Ctype_struct_CUctxCreateParams_st, _Ctype_uint, _Ctype_CUdevice)
```

**Root Cause:** The default mumax/3 v3.9.3 uses the CUDA 11.x API. CUDA 12.x changed the `cuCtxCreate` function signature.

**Solutions:**

#### Solution 1: Use mumax/3 v3.11.1 (Supports CUDA 12.9+)

mumax/3 v3.11.1 supports CUDA 12.x but requires manual setup due to Go module versioning issues.

1. **Find the v3.11.1 commit hash:**
   ```bash
   # Clone mumax repository
   git clone --branch=v3.11.1 --depth=1 https://github.com/mumax/3.git /tmp/mumax3
   cd /tmp/mumax3
   git rev-parse HEAD  # This gives you the commit hash
   ```

2. **Update your go.mod file:**
   
   Add this replace directive at the bottom of `go.mod`:
   ```go
   replace github.com/mumax/3 => github.com/mumax/3 v0.0.0-20241001120000-<commit-hash-from-step-1>
   ```
   
   The pseudo-version format is: `v0.0.0-YYYYMMDDHHMMSS-<12-char-commit-hash>`

3. **Rebuild:**
   ```bash
   go clean -modcache
   go mod tidy
   go build -tags cuda -o go-battleclank-cuda .
   ```

#### Solution 2: Downgrade to CUDA 11.8

If you prefer a simpler setup:

1. **Remove CUDA 12.x:**
   ```bash
   sudo apt-get remove --purge 'cuda-*' 'nvidia-cuda-*'
   sudo apt-get autoremove
   ```

2. **Install CUDA 11.8:**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get install cuda-toolkit-11-8
   ```

3. **Set environment variables:**
   ```bash
   export PATH=/usr/local/cuda-11.8/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
   
   # Add to ~/.bashrc for persistence
   echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Rebuild:**
   ```bash
   go clean -modcache
   go build -tags cuda -o go-battleclank-cuda .
   ```

#### Solution 3: Use CPU-only build
```bash
go build -o go-battleclank .
./go-battleclank --enable-gpu  # Will use CPU fallback
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
