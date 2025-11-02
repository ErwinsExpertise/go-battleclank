package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	
	"github.com/ErwinsExpertise/go-battleclank/gpu"
)

// Build information injected by GoReleaser
var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

func main() {
	// Check for version flag at any position in arguments (before parsing)
	for _, arg := range os.Args[1:] {
		if arg == "--version" || arg == "-v" {
			fmt.Printf("go-battleclank %s\n", version)
			fmt.Printf("Commit: %s\n", commit)
			fmt.Printf("Built: %s\n", date)
			return
		}
	}

	// Define CLI flags
	configPath := flag.String("config", "", "Path to configuration file (overrides BATTLESNAKE_CONFIG env var)")
	enableGPU := flag.Bool("enable-gpu", false, "Enable GPU acceleration for MCTS and other algorithms (requires CUDA)")
	flag.Parse()

	// Set config path via environment variable if specified via CLI
	if *configPath != "" {
		os.Setenv("BATTLESNAKE_CONFIG", *configPath)
	}

	// Initialize GPU if enabled
	if *enableGPU {
		gpu.SetEnabled(true)
		if err := gpu.Initialize(); err != nil {
			log.Printf("GPU initialization failed: %v (will use CPU fallback)", err)
		} else {
			log.Printf("GPU acceleration enabled: %s", gpu.DeviceName)
			defer gpu.Cleanup()
		}
	} else {
		log.Println("GPU acceleration disabled (use --enable-gpu to enable)")
	}

	RunServer()
}
