package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/ErwinsExpertise/go-battleclank/config"
)

// GetConfig is a helper to access config package
func GetConfig() *config.Config {
	return config.GetConfig()
}

// HTTP Handlers

// HandleIndex responds to GET / with snake info
func HandleIndex(w http.ResponseWriter, r *http.Request) {
	response := info()

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		log.Printf("ERROR: Failed to encode info response, %s", err)
	}
}

// HandleStart responds to POST /start at the beginning of a game
func HandleStart(w http.ResponseWriter, r *http.Request) {
	state := GameState{}
	err := json.NewDecoder(r.Body).Decode(&state)
	if err != nil {
		log.Printf("ERROR: Failed to decode start json, %s", err)
		return
	}

	// Use refactored logic by default unless explicitly disabled
	useLegacy := os.Getenv("USE_LEGACY") == "true"
	if useLegacy {
		start(state)
	} else {
		startRefactored(state)
	}

	// Nothing to respond with here
	w.WriteHeader(http.StatusOK)
}

// HandleMove responds to POST /move on each turn
func HandleMove(w http.ResponseWriter, r *http.Request) {
	state := GameState{}
	err := json.NewDecoder(r.Body).Decode(&state)
	if err != nil {
		log.Printf("ERROR: Failed to decode move json, %s", err)
		return
	}

	response := moveRefactored(state)

	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(response)
	if err != nil {
		log.Printf("ERROR: Failed to encode move response, %s", err)
		return
	}
}

// HandleEnd responds to POST /end when the game is over
func HandleEnd(w http.ResponseWriter, r *http.Request) {
	state := GameState{}
	err := json.NewDecoder(r.Body).Decode(&state)
	if err != nil {
		log.Printf("ERROR: Failed to decode end json, %s", err)
		return
	}

	endRefactored(state)

	w.WriteHeader(http.StatusOK)
}

// Middleware

const ServerID = "go-battleclank/1.0.0"

func withServerID(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Server", ServerID)
		next(w, r)
	}
}

// HandleReloadConfig responds to POST /reload-config to reload configuration
// Note: This endpoint allows runtime config reload, but it's recommended to restart
// the server for training to ensure all strategy instances use the new config
func HandleReloadConfig(w http.ResponseWriter, r *http.Request) {
	log.Println("Received config reload request")

	err := config.ReloadConfig()
	if err != nil {
		log.Printf("Error reloading config: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "error",
			"message": "Failed to reload config: " + err.Error(),
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "ok",
		"message": "Config reloaded successfully. Note: Existing strategy instances will continue using old config until next move.",
	})
}

// RunServer starts the Battlesnake HTTP server
func RunServer() {
	// Load config on startup to verify it's accessible and log settings
	_ = GetConfig()

	port := os.Getenv("PORT")
	if len(port) == 0 {
		port = "8000"
	}

	http.HandleFunc("/", withServerID(HandleIndex))
	http.HandleFunc("/start", withServerID(HandleStart))
	http.HandleFunc("/move", withServerID(HandleMove))
	http.HandleFunc("/end", withServerID(HandleEnd))
	http.HandleFunc("/reload-config", withServerID(HandleReloadConfig))

	log.Printf("Running Battlesnake at http://0.0.0.0:%s...\n", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
