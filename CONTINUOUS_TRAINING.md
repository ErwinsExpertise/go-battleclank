# 24/7 Continuous Training System

## Overview

Automated neural network training system that runs continuously without manual intervention. Automatically manages checkpoints, recovery, and improvement tracking.

## Features

- **24/7 Operation**: Runs indefinitely, exploring weight configurations
- **Auto-Checkpoint**: Saves state every 10 iterations
- **Auto-Recovery**: Resumes from last checkpoint on crash/restart
- **Graceful Shutdown**: Ctrl+C saves state before exiting
- **Auto-Restart**: Bash script restarts training on unexpected crashes
- **Progress Tracking**: Comprehensive logging of all iterations
- **Best Config Tracking**: Always maintains best configuration found

## Quick Start

### Simple Start (Manual Monitoring)

```bash
cd /home/runner/work/go-battleclank/go-battleclank
python3 tools/continuous_training.py
```

### Production Start (Auto-Restart + Logging)

```bash
cd /home/runner/work/go-battleclank/go-battleclank
./tools/start_training.sh
```

### Background Training (Screen/Tmux)

```bash
# Using screen
screen -S battlesnake-training
./tools/start_training.sh
# Press Ctrl+A then D to detach

# Reattach later
screen -r battlesnake-training

# Using tmux
tmux new -s battlesnake-training
./tools/start_training.sh
# Press Ctrl+B then D to detach

# Reattach later
tmux attach -t battlesnake-training
```

### Systemd Service (Linux Production)

Create `/etc/systemd/system/battlesnake-training.service`:

```ini
[Unit]
Description=Battlesnake 24/7 Continuous Training
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/runner/work/go-battleclank/go-battleclank
ExecStart=/home/runner/work/go-battleclank/go-battleclank/tools/start_training.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable battlesnake-training
sudo systemctl start battlesnake-training
sudo systemctl status battlesnake-training
```

View logs:
```bash
sudo journalctl -u battlesnake-training -f
```

## Configuration

Edit command-line arguments in `tools/start_training.sh` or pass directly:

```bash
python3 tools/continuous_training.py \
    --games 50 \                    # Games per test (default: 30)
    --checkpoint-interval 5 \       # Save every N iterations (default: 10)
    --max-iterations 1000 \         # Stop after N iterations (default: unlimited)
    --min-improvement 0.002         # Min improvement to keep (default: 0.001 = 0.1%)
```

## How It Works

1. **Initialization**
   - Load checkpoint if exists, otherwise start fresh
   - Establish baseline win rate (50 games)
   - Setup signal handlers for graceful shutdown

2. **Training Loop** (per iteration)
   - Load current best configuration from `config.yaml`
   - Generate candidate config (random 15% perturbation on 3-5 weights)
   - Save candidate to `config.yaml`
   - Rebuild Go snake
   - Run N-game benchmark
   - Compare to best win rate
   - If improved: Keep config, update best
   - If not improved: Restore previous config
   - Log results to `training_log.jsonl`

3. **Checkpointing**
   - Save state every 10 iterations (configurable)
   - Includes: iteration count, best config, best win rate, total games
   - Enables resume from any point

4. **Graceful Shutdown**
   - Ctrl+C triggers signal handler
   - Saves checkpoint before exit
   - Training can resume exactly where it left off

## File Structure

```
go-battleclank/
├── config.yaml                        # Best configuration (updated automatically)
├── nn_training_results/               # All training data
│   ├── checkpoint.json                # Resume state
│   ├── training_log.jsonl             # Full iteration history (append-only)
│   └── training_YYYYMMDD_HHMMSS.log  # Console output
└── tools/
    ├── continuous_training.py         # Main training script
    └── start_training.sh              # Auto-restart wrapper
```

## Monitoring

### Check Current Status

```bash
# View latest iterations
tail -20 nn_training_results/training_log.jsonl | jq -r '[.iteration, .win_rate, .best_win_rate] | @tsv'

# View best configuration
cat config.yaml

# Check checkpoint status
cat nn_training_results/checkpoint.json | jq
```

### Live Monitoring

```bash
# Watch training log
tail -f nn_training_results/training_*.log

# Watch iteration results
tail -f nn_training_results/training_log.jsonl | jq -C
```

### Statistics

```bash
# Count total iterations
wc -l nn_training_results/training_log.jsonl

# Find best iteration
cat nn_training_results/training_log.jsonl | jq -s 'max_by(.win_rate)'

# Count improvements
cat nn_training_results/training_log.jsonl | jq -s '[.[] | select(.improvement > 0)] | length'

# Average win rate
cat nn_training_results/training_log.jsonl | jq -s '[.[].win_rate] | add / length'
```

## Expected Performance

**Current Baseline**: 47% win rate

**Training Speed**:
- ~30 sec/game (depends on hardware)
- ~15 min/iteration (30 games)
- ~4 hours for 16 iterations (checkpoint)

**Expected Progress**:
- First 100 iterations: Find local improvements (+2-5%)
- 100-500 iterations: Explore parameter space systematically
- 500-1000 iterations: Fine-tuning around best configurations
- **Target**: 52-55% win rate (+5-8 points)

**Stagnation Detection**:
- Warning after 100 iterations without improvement
- Consider stopping or adjusting perturbation magnitude
- May indicate local optimum reached

## Troubleshooting

### Training Won't Start

```bash
# Check dependencies
pip install -r requirements.txt

# Verify setup
./tools/quick_setup.sh

# Check config file exists
ls -la config.yaml
```

### Training Crashes

- Auto-restart is enabled in `start_training.sh`
- Check logs: `tail -50 nn_training_results/training_*.log`
- Resume manually: `python3 tools/continuous_training.py`

### Checkpoint Corruption

```bash
# Backup old checkpoint
mv nn_training_results/checkpoint.json nn_training_results/checkpoint.json.backup

# Restart fresh (will re-establish baseline)
python3 tools/continuous_training.py
```

### Out of Disk Space

```bash
# Training logs can grow large
# Compress old logs
gzip nn_training_results/training_*.log

# Keep only recent iterations
tail -1000 nn_training_results/training_log.jsonl > nn_training_results/training_log_recent.jsonl
mv nn_training_results/training_log_recent.jsonl nn_training_results/training_log.jsonl
```

## Stopping Training

### Graceful Stop

```bash
# Press Ctrl+C
# Waits for current iteration to finish
# Saves checkpoint automatically
```

### Force Stop

```bash
# Find process
ps aux | grep continuous_training

# Kill process (will save checkpoint)
kill -TERM <PID>

# If frozen
kill -KILL <PID>  # Last resort, checkpoint may not save
```

### Systemd

```bash
sudo systemctl stop battlesnake-training
```

## Resuming Training

Training automatically resumes from last checkpoint:

```bash
./tools/start_training.sh
# Output: "✓ Resumed from checkpoint: iteration 42, best win rate 51.2%"
```

## Advanced Usage

### Custom Perturbation Strategy

Edit `continuous_training.py`:

```python
def random_perturbation(self, config, magnitude=0.1):
    # Customize which weights to adjust
    # Customize adjustment magnitude
    # Customize selection strategy
```

### Integration with Other Systems

```python
# Import as library
from tools.continuous_training import ContinuousTrainer

trainer = ContinuousTrainer(games_per_iteration=50)
trainer.train()
```

### Cloud Deployment

```bash
# AWS EC2, Google Compute, Azure VM
# 1. Clone repository
# 2. Setup dependencies
# 3. Run in tmux/screen
# 4. Monitor remotely

# Example with tmux
tmux new -s training
./tools/start_training.sh
# Detach: Ctrl+B then D
# Close SSH, training continues
```

## Cost Analysis

**Time Investment**:
- Setup: 5 minutes
- Initial baseline: 25 minutes (50 games)
- Per iteration: 15 minutes (30 games)
- 1000 iterations: ~10.4 days continuous

**Compute Cost** (AWS p3.2xlarge with GPU):
- ~$3/hour
- 1000 iterations = ~$750
- Expected gain: +5-8% win rate

**Value**:
- Systematic exploration of parameter space
- No manual intervention required
- Finds improvements humans miss
- Reproducible and documented

## Comparison to Manual Tuning

| Aspect | Manual Tuning | 24/7 Training |
|--------|--------------|---------------|
| Time | 40+ hours | 5 min setup |
| Coverage | 16 configs | 1000+ configs |
| Systematic | No | Yes |
| Reproducible | Partial | Full |
| Best Result | 47% | 52-55% expected |
| Human Effort | High | Minimal |

## Future Enhancements

Planned improvements:
- [ ] Parallel benchmark execution (multiple snakes)
- [ ] Bayesian optimization for smarter perturbations
- [ ] Multi-objective optimization (win rate + game length)
- [ ] Tournament mode (test vs multiple opponents)
- [ ] Reinforcement learning integration
- [ ] Web dashboard for monitoring

## Support

Issues or questions:
1. Check logs: `nn_training_results/training_*.log`
2. Check checkpoint: `nn_training_results/checkpoint.json`
3. Verify dependencies: `pip list`
4. Review configuration: `config.yaml`

## Summary

The 24/7 continuous training system provides:
- ✅ Fully automated weight optimization
- ✅ No manual intervention required
- ✅ Automatic checkpoint and recovery
- ✅ Comprehensive logging and monitoring
- ✅ Production-ready with auto-restart
- ✅ Systematically explores vast parameter space
- ✅ Expected +5-8% improvement over manual tuning

**To start**: `./tools/start_training.sh`
**To stop**: Press Ctrl+C (saves checkpoint)
**To resume**: Run same command again
