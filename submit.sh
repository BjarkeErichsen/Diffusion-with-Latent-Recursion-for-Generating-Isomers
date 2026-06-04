#!/bin/bash

# 1. Extract the run name from the jobscript for better organization
WANDB_NAME=$(grep "logger.wandb.name=" custom/jobscript.sh | sed -E 's/.*logger\.wandb\.name="([^"]*)".*/\1/' | head -n 1 | sed 's/[^a-zA-Z0-9_-]/_/g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ -z "$WANDB_NAME" ]; then
    SNAPSHOT_DIR="snapshots/run_$TIMESTAMP"
else
    # Optionally truncate if it's too long, but let's keep it for now
    SNAPSHOT_DIR="snapshots/${WANDB_NAME}_run_$TIMESTAMP"
fi
mkdir -p "$SNAPSHOT_DIR"

# 2. Copy your code and configs (the things you edit)
cp -r src_gmmm custom configs "$SNAPSHOT_DIR/"

# 3. Symlink the heavy/static directories so you don't waste disk space
ln -s "$PWD/.venv" "$SNAPSHOT_DIR/.venv"
ln -s "$PWD/data" "$SNAPSHOT_DIR/data"
# If your output needs to go to a shared folder, symlink it too
# ln -s "$PWD/output" "$SNAPSHOT_DIR/output" 

# 4. Move into the snapshot and submit
cd "$SNAPSHOT_DIR" || exit
bsub < custom/jobscript.sh
echo "Job submitted from $SNAPSHOT_DIR"
cd - > /dev/null