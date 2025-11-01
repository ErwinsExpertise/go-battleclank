#!/bin/bash
# Test script to verify config changes are applied without rebuild

set -e

echo "=== Config Reload Test ==="
echo

# Build the battlesnake
echo "1. Building battlesnake..."
go build -o battlesnake
echo "   ✓ Build complete"
echo

# Backup original config
echo "2. Backing up original config.yaml..."
cp config.yaml config.yaml.backup
echo "   ✓ Backup saved"
echo

# Start the server in background
echo "3. Starting battlesnake server..."
./battlesnake > server.log 2>&1 &
SERVER_PID=$!
sleep 2
echo "   ✓ Server started (PID: $SERVER_PID)"
echo

# Check the config was loaded
echo "4. Checking initial config load..."
grep "Configuration loaded successfully" server.log
grep "Space weight:" server.log | head -1
echo

# Modify config
echo "5. Modifying config.yaml (changing space weight)..."
# Read current space value
CURRENT_SPACE=$(grep "  space:" config.yaml | awk '{print $2}')
NEW_SPACE="10.0"
sed -i "s/  space: ${CURRENT_SPACE}/  space: ${NEW_SPACE}/" config.yaml
echo "   ✓ Config modified (space: ${CURRENT_SPACE} → ${NEW_SPACE})"
echo

# Kill and restart server
echo "6. Restarting server to apply new config..."
kill $SERVER_PID 2>/dev/null || true
sleep 1
./battlesnake > server2.log 2>&1 &
SERVER_PID=$!
sleep 2
echo "   ✓ Server restarted (PID: $SERVER_PID)"
echo

# Verify new config was loaded
echo "7. Verifying new config values loaded..."
if grep -q "Space weight: ${NEW_SPACE}" server2.log; then
    echo "   ✓ SUCCESS: New config values loaded!"
    echo "   Space weight changed from ${CURRENT_SPACE} to ${NEW_SPACE}"
else
    echo "   ✗ FAILED: Config not updated"
    grep "Space weight:" server2.log | head -1
    exit 1
fi
echo

# Cleanup
echo "8. Cleanup..."
kill $SERVER_PID 2>/dev/null || true
cp config.yaml.backup config.yaml
rm -f config.yaml.backup server.log server2.log
echo "   ✓ Original config restored"
echo

echo "=== Test Complete: Config reload works correctly ==="
