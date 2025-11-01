package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
)

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

	// Use refactored logic by default unless explicitly disabled
	useLegacy := os.Getenv("USE_LEGACY") == "true"
	
	var response BattlesnakeMoveResponse
	if useLegacy {
		response = move(state)
	} else {
		response = moveRefactored(state)
	}

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

	// Use refactored logic by default unless explicitly disabled
	useLegacy := os.Getenv("USE_LEGACY") == "true"
	if useLegacy {
		end(state)
	} else {
		endRefactored(state)
	}

	// Nothing to respond with here
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

// RunServer starts the Battlesnake HTTP server
func RunServer() {
	port := os.Getenv("PORT")
	if len(port) == 0 {
		port = "8000"
	}

	http.HandleFunc("/", withServerID(HandleIndex))
	http.HandleFunc("/start", withServerID(HandleStart))
	http.HandleFunc("/move", withServerID(HandleMove))
	http.HandleFunc("/end", withServerID(HandleEnd))

	log.Printf("Running Battlesnake at http://0.0.0.0:%s...\n", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
