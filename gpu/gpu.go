//go:build !cuda
// +build !cuda

package gpu

import (
	"fmt"
	"log"
)

var (
	// GPUEnabled indicates whether GPU acceleration is enabled via CLI flag
	GPUEnabled bool
	
	// GPUAvailable indicates whether GPU hardware/drivers are actually available
	GPUAvailable bool
	
	// DeviceCount is the number of GPU devices detected
	DeviceCount int
	
	// DeviceName is the name of the primary GPU device
	DeviceName string
)

// Initialize checks for GPU availability and initializes GPU acceleration
// This function is safe to call even when CUDA is not available - it will
// gracefully fallback to CPU mode
// 
// NOTE: This is the non-CUDA fallback version. To enable CUDA, build with: go build -tags cuda
func Initialize() error {
	if !GPUEnabled {
		log.Println("GPU acceleration disabled (use --enable-gpu to enable)")
		GPUAvailable = false
		return fmt.Errorf("GPU not enabled via CLI flag")
	}

	log.Println("Binary built without CUDA support. To enable GPU acceleration, rebuild with: go build -tags cuda")
	log.Println("Using CPU fallback")
	GPUAvailable = false
	return fmt.Errorf("binary not built with CUDA support (use -tags cuda)")
}

// Cleanup releases GPU resources
func Cleanup() {
	// No-op in non-CUDA build
}

// SetEnabled sets whether GPU acceleration should be enabled
func SetEnabled(enabled bool) {
	GPUEnabled = enabled
}

// IsAvailable returns true if GPU acceleration is both enabled and available
func IsAvailable() bool {
	return GPUEnabled && GPUAvailable
}

// WithDevice runs a function on a specific GPU device
func WithDevice(deviceID int, fn func() error) error {
	return fmt.Errorf("GPU not available in non-CUDA build")
}
