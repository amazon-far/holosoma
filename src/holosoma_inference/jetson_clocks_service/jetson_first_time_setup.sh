#!/bin/bash
# Install and enable jetson_clocks.service systemd service & set other performance

# Show commands
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_FILE="$SCRIPT_DIR/jetson_clocks.service"
DEST_FILE="/etc/systemd/system/jetson_clocks.service"

# Run with sudo
if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run this script with sudo:"
  echo "   sudo $0"
  exit 1
fi

echo "📂 Copying $SRC_FILE → $DEST_FILE"
cp "$SRC_FILE" "$DEST_FILE"

echo "🔁 Reloading systemd daemon..."
systemctl daemon-reload

echo "✅ Enabling jetson_clocks.service..."
systemctl enable jetson_clocks.service

echo "🚀 Starting jetson_clocks.service now..."
systemctl start jetson_clocks.service

echo "🧠 Checking status of the service"
systemctl status jetson_clocks.service --no-pager

# Use all cores
nvpmodel -m 0

echo "🎉 Done! The device will restart now to apply changes"
sleep 3
reboot
