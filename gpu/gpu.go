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
func Initialize() error {
	if !GPUEnabled {
		log.Println("GPU acceleration disabled (use --enable-gpu to enable)")
		GPUAvailable = false
		return fmt.Errorf("GPU not enabled via CLI flag")
	}

	// Attempt to initialize CUDA
	// In this initial implementation, we'll use CPU fallback
	// Future versions will use actual CUDA bindings (e.g., mumax/3/cuda)
	
	defer func() {
		if r := recover(); r != nil {
			log.Printf("GPU initialization failed: %v (using CPU fallback)", r)
			GPUAvailable = false
		}
	}()
	
	// Try to detect GPU - this will be implemented with actual CUDA calls later
	// For now, we simulate the check
	if !checkCUDAAvailable() {
		log.Println("CUDA not available, using CPU fallback")
		GPUAvailable = false
		return fmt.Errorf("CUDA not available")
	}
	
	// Simulate device detection
	DeviceCount = detectDeviceCount()
	if DeviceCount == 0 {
		log.Println("No CUDA devices found, using CPU fallback")
		GPUAvailable = false
		return fmt.Errorf("no CUDA devices found")
	}
	
	// Select device 0 by default
	DeviceName = getDeviceName(0)
	
	log.Printf("GPU initialized: %s (device 0 of %d)", DeviceName, DeviceCount)
	GPUAvailable = true
	
	return nil
}

// Cleanup releases GPU resources
func Cleanup() {
	if GPUAvailable {
		log.Println("Cleaning up GPU resources")
		// Future: cuda.Recycle()
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

// --- Internal helper functions for CPU fallback simulation ---

// checkCUDAAvailable checks if CUDA is available
// Future: will use cuda.Available() from mumax/3/cuda
func checkCUDAAvailable() bool {
	// For now, always return false to use CPU fallback
	// When CUDA bindings are added, this will do actual detection
	return false
}

// detectDeviceCount detects the number of CUDA devices
// Future: will use cuda.DeviceCount()
func detectDeviceCount() int {
	// For now, return 0 to indicate no devices
	return 0
}

// getDeviceName gets the name of a specific device
// Future: will use cuda.DeviceName(deviceID)
func getDeviceName(deviceID int) string {
	// Placeholder
	return fmt.Sprintf("CUDA Device %d", deviceID)
}
