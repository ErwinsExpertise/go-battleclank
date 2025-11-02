#!/bin/bash
# Example: LLM-Enhanced Training Usage Examples

set -e

echo "=========================================="
echo "LLM-Enhanced Training Examples"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "config.yaml" ]; then
    echo "Error: Run this from the repository root"
    exit 1
fi

# Function to show example
show_example() {
    echo "Example: $1"
    echo "Command: $2"
    echo "Description: $3"
    echo ""
}

# Example 1: Basic LLM training
show_example "Basic LLM Training" \
    "python3 tools/continuous_training.py --use-llm --max-iterations 10" \
    "Run 10 iterations with LLM guidance, testing 30 games per iteration"

# Example 2: Full intelligence mode
show_example "Full Intelligence (LLM + Neural Net)" \
    "python3 tools/continuous_training.py --use-llm --use-neural-net --max-iterations 50" \
    "Run 50 iterations with both LLM and neural network enabled"

# Example 3: Fast testing without LLM
show_example "Fast Testing (No LLM)" \
    "python3 tools/continuous_training.py --no-llm --games 10 --max-iterations 5" \
    "Quick test with 5 iterations, 10 games each, no LLM overhead"

# Example 4: Production settings for A100
show_example "Production A100 Settings" \
    "python3 tools/continuous_training.py --use-llm --use-neural-net --games 50 --checkpoint-interval 5" \
    "Production training with 50 games per test, frequent checkpoints"

# Example 5: Neural network only
show_example "Neural Network Only" \
    "python3 tools/continuous_training.py --no-llm --use-neural-net --max-iterations 20" \
    "Use neural network pattern recognition without LLM (faster)"

# Example 6: Check config coverage
show_example "Verify Config Coverage" \
    "python3 tools/test_config_coverage.py" \
    "Test that all 36 config parameters are accessible"

echo "=========================================="
echo "Interactive Examples"
echo "=========================================="
echo ""

# Offer to run examples
read -p "Run Example 1 (Basic LLM Training - 10 iterations)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting Basic LLM Training..."
    python3 tools/continuous_training.py --use-llm --max-iterations 10 --games 10
fi

read -p "Run Config Coverage Test? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running Config Coverage Test..."
    python3 tools/test_config_coverage.py
fi

echo ""
echo "=========================================="
echo "For more details, see:"
echo "  - LLM_TRAINING_GUIDE.md"
echo "  - CONTINUOUS_TRAINING.md"
echo "=========================================="
