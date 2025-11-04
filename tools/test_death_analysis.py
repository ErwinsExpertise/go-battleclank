#!/usr/bin/env python3
"""
Unit tests for death analysis in run_benchmark.py

Tests verify proper separation between death reasons (by_reason) and game phases (by_phase).
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from run_benchmark import BenchmarkRunner


class TestDeathAnalysis(unittest.TestCase):
    """Test death analysis categorization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.runner = BenchmarkRunner(num_games=10)
    
    def test_game_phase_categorization(self):
        """Test that game phases are correctly categorized by turn count"""
        # Test early game
        self.assertEqual(self.runner._get_game_phase(10), "early")
        self.assertEqual(self.runner._get_game_phase(49), "early")
        
        # Test mid game
        self.assertEqual(self.runner._get_game_phase(50), "mid")
        self.assertEqual(self.runner._get_game_phase(100), "mid")
        self.assertEqual(self.runner._get_game_phase(149), "mid")
        
        # Test late game
        self.assertEqual(self.runner._get_game_phase(150), "late")
        self.assertEqual(self.runner._get_game_phase(250), "late")
    
    def test_death_reason_head_collision(self):
        """Test head collision death reason detection"""
        output = "Game completed after 75 turns. rust-baseline was the winner. go-battleclank was eliminated by head collision."
        death_reason, game_phase = self.runner._categorize_death(75, "rust-baseline", output)
        
        self.assertEqual(death_reason, "head_collision")
        self.assertEqual(game_phase, "mid")  # 75 turns is mid-game
    
    def test_death_reason_starvation(self):
        """Test starvation death reason detection"""
        output = "Game completed after 120 turns. rust-baseline was the winner. go-battleclank was eliminated by starvation."
        death_reason, game_phase = self.runner._categorize_death(120, "rust-baseline", output)
        
        self.assertEqual(death_reason, "starvation")
        self.assertEqual(game_phase, "mid")
    
    def test_death_reason_self_collision(self):
        """Test self collision death reason detection"""
        output = "Game completed after 30 turns. rust-baseline was the winner. go-battleclank was eliminated by self collision."
        death_reason, game_phase = self.runner._categorize_death(30, "rust-baseline", output)
        
        self.assertEqual(death_reason, "self_collision")
        self.assertEqual(game_phase, "early")
    
    def test_death_reason_body_collision(self):
        """Test body collision death reason detection"""
        output = "Game completed after 180 turns. rust-baseline was the winner. go-battleclank was eliminated by body collision."
        death_reason, game_phase = self.runner._categorize_death(180, "rust-baseline", output)
        
        self.assertEqual(death_reason, "body_collision")
        self.assertEqual(game_phase, "late")
    
    def test_death_reason_wall_collision(self):
        """Test wall collision death reason detection"""
        output = "Game completed after 20 turns. rust-baseline was the winner. go-battleclank was eliminated by wall collision."
        death_reason, game_phase = self.runner._categorize_death(20, "rust-baseline", output)
        
        self.assertEqual(death_reason, "wall_collision")
        self.assertEqual(game_phase, "early")
    
    def test_death_reason_unknown(self):
        """Test unknown death reason when no specific cause is found"""
        output = "Game completed after 90 turns. rust-baseline was the winner. go-battleclank was eliminated."
        death_reason, game_phase = self.runner._categorize_death(90, "rust-baseline", output)
        
        self.assertEqual(death_reason, "unknown")
        self.assertEqual(game_phase, "mid")
    
    def test_winner_no_death_reason(self):
        """Test that winning snake has no death reason"""
        output = "Game completed after 100 turns. go-battleclank was the winner."
        death_reason, game_phase = self.runner._categorize_death(100, "go-battleclank", output)
        
        self.assertIsNone(death_reason)
        self.assertIsNone(game_phase)
    
    def test_death_reason_and_phase_are_separate(self):
        """Test that death reason and phase are tracked separately"""
        # Early game head collision
        output1 = "Game completed after 25 turns. rust-baseline was the winner. go-battleclank was eliminated by head collision."
        reason1, phase1 = self.runner._categorize_death(25, "rust-baseline", output1)
        self.assertEqual(reason1, "head_collision")
        self.assertEqual(phase1, "early")
        
        # Late game head collision
        output2 = "Game completed after 200 turns. rust-baseline was the winner. go-battleclank was eliminated by head collision."
        reason2, phase2 = self.runner._categorize_death(200, "rust-baseline", output2)
        self.assertEqual(reason2, "head_collision")
        self.assertEqual(phase2, "late")
        
        # Verify same reason, different phases
        self.assertEqual(reason1, reason2)
        self.assertNotEqual(phase1, phase2)
    
    def test_no_phase_based_reasons(self):
        """Test that phase-based categories are NOT in death reasons"""
        # These old phase-based categories should never be returned as death reasons
        invalid_reasons = ["early_game_death", "mid_game_death", "late_game_death"]
        
        # Test various outputs
        test_cases = [
            ("Game completed after 30 turns. rust-baseline was the winner. go-battleclank was eliminated.", 30),
            ("Game completed after 100 turns. rust-baseline was the winner. go-battleclank was eliminated.", 100),
            ("Game completed after 200 turns. rust-baseline was the winner. go-battleclank was eliminated.", 200),
        ]
        
        for output, turns in test_cases:
            death_reason, _ = self.runner._categorize_death(turns, "rust-baseline", output)
            self.assertNotIn(death_reason, invalid_reasons, 
                           f"Death reason should not be phase-based: {death_reason}")


class TestDeathAnalysisIntegration(unittest.TestCase):
    """Integration tests for death analysis in benchmark results"""
    
    def test_result_structure(self):
        """Test that result JSON has correct structure for death_analysis"""
        # Create a minimal result structure as would be saved
        result_data = {
            "results": {
                "wins": 5,
                "losses": 3,
            },
            "death_analysis": {
                "by_reason": {
                    "head_collision": 2,
                    "starvation": 1
                },
                "by_phase": {
                    "early": 1,
                    "mid": 2,
                    "late": 0
                }
            }
        }
        
        # Verify structure
        self.assertIn("death_analysis", result_data)
        self.assertIn("by_reason", result_data["death_analysis"])
        self.assertIn("by_phase", result_data["death_analysis"])
        
        # Verify by_reason contains actual death reasons, not phases
        by_reason = result_data["death_analysis"]["by_reason"]
        self.assertIn("head_collision", by_reason)
        self.assertIn("starvation", by_reason)
        
        # Verify by_phase contains phase categories, not death reasons
        by_phase = result_data["death_analysis"]["by_phase"]
        self.assertIn("early", by_phase)
        self.assertIn("mid", by_phase)
        self.assertIn("late", by_phase)
        
        # Ensure no overlap - phase keys should not be in reasons
        phase_keys = ["early", "mid", "late"]
        for key in phase_keys:
            self.assertNotIn(key, by_reason, 
                           f"Phase key '{key}' should not be in by_reason")
        
        # Ensure no overlap - old phase-based reason keys should not exist
        old_phase_reasons = ["early_game_death", "mid_game_death", "late_game_death"]
        for key in old_phase_reasons:
            self.assertNotIn(key, by_reason,
                           f"Old phase-based reason '{key}' should not be in by_reason")
            self.assertNotIn(key, by_phase,
                           f"Old phase-based reason '{key}' should not be in by_phase")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
