package main

import (
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
	// Check for version flag at any position in arguments
	for _, arg := range os.Args[1:] {
		if arg == "--version" || arg == "-v" {
			fmt.Printf("go-battleclank %s\n", version)
			fmt.Printf("Commit: %s\n", commit)
			fmt.Printf("Built: %s\n", date)
			return
		}
	}

	RunServer()
}
