package gpu

import (
	"testing"
)

func TestGPUInitialization(t *testing.T) {
	// Test with GPU disabled
	SetEnabled(false)
	err := Initialize()
	if err == nil {
		t.Error("Expected error when GPU is disabled, got nil")
	}
	if GPUAvailable {
		t.Error("GPUAvailable should be false when disabled")
	}
	
	// Test with GPU enabled (will fallback to CPU since CUDA not available)
	SetEnabled(true)
	err = Initialize()
	if err == nil {
		t.Log("GPU initialized successfully (CUDA available)")
	} else {
		t.Logf("GPU initialization failed as expected: %v (CUDA not available, using CPU fallback)", err)
	}
	
	// GPU should not be available in test environment (no CUDA)
	if GPUAvailable {
		t.Log("GPU is available, will use GPU acceleration")
		Cleanup()
	} else {
		t.Log("GPU not available (expected), will use CPU fallback")
	}
}

func TestIsAvailable(t *testing.T) {
	// Test when disabled
	SetEnabled(false)
	if IsAvailable() {
		t.Error("IsAvailable should return false when GPU is disabled")
	}
	
	// Test when enabled but not available (no CUDA in test env)
	SetEnabled(true)
	Initialize()
	available := IsAvailable()
	
	// Should be false in test environment without CUDA
	if available {
		t.Log("GPU is available in test environment")
	} else {
		t.Log("GPU not available in test environment (expected)")
	}
}

func TestSetEnabled(t *testing.T) {
	SetEnabled(true)
	if !GPUEnabled {
		t.Error("SetEnabled(true) should set GPUEnabled to true")
	}
	
	SetEnabled(false)
	if GPUEnabled {
		t.Error("SetEnabled(false) should set GPUEnabled to false")
	}
}

func TestCleanup(t *testing.T) {
	// Test cleanup doesn't panic
	SetEnabled(true)
	Initialize()
	Cleanup()
	
	SetEnabled(false)
	Cleanup()
}
