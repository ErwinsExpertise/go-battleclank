//go:build cuda
// +build cuda

package gpu

import (
	"fmt"
	"log"
	
	"github.com/mumax/3/cuda"
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
func Initialize() error {
	if !GPUEnabled {
		log.Println("GPU acceleration disabled (use --enable-gpu to enable)")
		GPUAvailable = false
		return fmt.Errorf("GPU not enabled via CLI flag")
	}

	// Attempt to initialize CUDA with panic recovery for graceful fallback
	defer func() {
		if r := recover(); r != nil {
			log.Printf("GPU initialization failed: %v (using CPU fallback)", r)
			GPUAvailable = false
		}
	}()
	
	// Check if CUDA is available
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
		log.Println("Cleaning up GPU resources")
		cuda.Recycle()
	}
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
