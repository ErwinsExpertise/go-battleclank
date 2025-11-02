package main

import (
	"flag"
	"fmt"
	"os"
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
	flag.Parse()

	// Set config path via environment variable if specified via CLI
	if *configPath != "" {
		os.Setenv("BATTLESNAKE_CONFIG", *configPath)
	}

	RunServer()
}
