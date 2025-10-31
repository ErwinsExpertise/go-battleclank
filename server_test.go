package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Test HandleIndex
func TestHandleIndex(t *testing.T) {
	req, err := http.NewRequest("GET", "/", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(HandleIndex)
	handler.ServeHTTP(rr, req)

	// Check status code
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("Handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}

	// Check content type
	contentType := rr.Header().Get("Content-Type")
	if contentType != "application/json" {
		t.Errorf("Handler returned wrong content type: got %v want %v", contentType, "application/json")
	}

	// Check response body can be decoded
	var response BattlesnakeInfoResponse
	err = json.NewDecoder(rr.Body).Decode(&response)
	if err != nil {
		t.Errorf("Failed to decode response: %v", err)
	}

	// Check required fields
	if response.APIVersion == "" {
		t.Error("APIVersion should not be empty")
	}
}

// Test HandleStart
func TestHandleStart(t *testing.T) {
	gameState := GameState{
		Game: Game{
			ID: "test-game",
		},
		Turn: 0,
		Board: Board{
			Width:  11,
			Height: 11,
		},
		You: Battlesnake{
			ID:     "test-snake",
			Health: 100,
		},
	}

	body, err := json.Marshal(gameState)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest("POST", "/start", bytes.NewBuffer(body))
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(HandleStart)
	handler.ServeHTTP(rr, req)

	// Check status code
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("Handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}
}

// Test HandleMove
func TestHandleMove(t *testing.T) {
	gameState := GameState{
		Game: Game{
			ID: "test-game",
		},
		Turn: 1,
		Board: Board{
			Width:  11,
			Height: 11,
			Snakes: []Battlesnake{
				{
					ID:     "test-snake",
					Health: 100,
					Length: 3,
					Head:   Coord{X: 5, Y: 5},
					Body: []Coord{
						{X: 5, Y: 5},
						{X: 5, Y: 4},
						{X: 5, Y: 3},
					},
				},
			},
		},
		You: Battlesnake{
			ID:     "test-snake",
			Health: 100,
			Length: 3,
			Head:   Coord{X: 5, Y: 5},
			Body: []Coord{
				{X: 5, Y: 5},
				{X: 5, Y: 4},
				{X: 5, Y: 3},
			},
		},
	}

	body, err := json.Marshal(gameState)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest("POST", "/move", bytes.NewBuffer(body))
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(HandleMove)
	handler.ServeHTTP(rr, req)

	// Check status code
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("Handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}

	// Check content type
	contentType := rr.Header().Get("Content-Type")
	if contentType != "application/json" {
		t.Errorf("Handler returned wrong content type: got %v want %v", contentType, "application/json")
	}

	// Check response body
	var response BattlesnakeMoveResponse
	err = json.NewDecoder(rr.Body).Decode(&response)
	if err != nil {
		t.Errorf("Failed to decode response: %v", err)
	}

	// Check move is valid
	validMoves := map[string]bool{
		"up":    true,
		"down":  true,
		"left":  true,
		"right": true,
	}
	if !validMoves[response.Move] {
		t.Errorf("Invalid move returned: %s", response.Move)
	}
}

// Test HandleEnd
func TestHandleEnd(t *testing.T) {
	gameState := GameState{
		Game: Game{
			ID: "test-game",
		},
		Turn: 100,
		Board: Board{
			Width:  11,
			Height: 11,
		},
		You: Battlesnake{
			ID:     "test-snake",
			Health: 0,
		},
	}

	body, err := json.Marshal(gameState)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest("POST", "/end", bytes.NewBuffer(body))
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(HandleEnd)
	handler.ServeHTTP(rr, req)

	// Check status code
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("Handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}
}

// Test HandleMove with invalid JSON
func TestHandleMove_InvalidJSON(t *testing.T) {
	req, err := http.NewRequest("POST", "/move", bytes.NewBufferString("invalid json"))
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(HandleMove)
	handler.ServeHTTP(rr, req)

	// Handler should not crash on invalid JSON
	// (it logs error but doesn't set special status code in current implementation)
}

// Test withServerID middleware
func TestWithServerID(t *testing.T) {
	testHandler := func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}

	req, err := http.NewRequest("GET", "/", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := withServerID(testHandler)
	handler.ServeHTTP(rr, req)

	// Check Server header is set
	serverHeader := rr.Header().Get("Server")
	if serverHeader != ServerID {
		t.Errorf("Expected Server header to be %s, got %s", ServerID, serverHeader)
	}
}
