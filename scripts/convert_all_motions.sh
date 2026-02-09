#!/bin/bash
# Convert all motion npz files to numpy 1.x compatible format

set -e  # Exit on error

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

source scripts/source_isaacsim_setup.sh

echo "==========================================="
echo "Converting all motion files to numpy 1.x"
echo "==========================================="

# Temporarily upgrade to numpy 2.x for conversion
echo "Installing numpy 2.x for conversion..."
pip install --upgrade "numpy>=2.0.0" --quiet

# Find all npz files except backups
FILES=$(find src/holosoma/holosoma/data/motions/h1_2 src/holosoma/holosoma/data/motions/statue_v0_1 -name "*.npz" -type f | grep -v "original_numpy2" | sort)

COUNT=0
TOTAL=$(echo "$FILES" | wc -l)

for FILE in $FILES; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] Converting: $FILE"

    # Backup original if not already backed up
    BACKUP="${FILE%.npz}_original_numpy2.npz"
    if [ ! -f "$BACKUP" ]; then
        echo "  Creating backup: $BACKUP"
        cp "$FILE" "$BACKUP"
    else
        echo "  Backup already exists: $BACKUP"
    fi

    # Convert
    python scripts/convert_motion_npz.py "$BACKUP" "$FILE"
done

# Restore numpy 1.x
echo ""
echo "==========================================="
echo "Restoring numpy 1.23.5..."
pip install "numpy==1.23.5" --quiet

echo ""
echo "✓ All conversions complete!"
echo "  Converted $COUNT files"
echo "  Numpy version: $(python -c 'import numpy; print(numpy.__version__)')"
