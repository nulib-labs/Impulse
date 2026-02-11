#!/bin/bash

# Directory to search in (defaults to current directory)
PARENT_DIR="lost_flows"

# Check if the parent directory exists
if [ ! -d "$PARENT_DIR" ]; then
  echo "Error: Directory '$PARENT_DIR' does not exist"
  exit 1
fi

# Loop over all subdirectories
for dir in "$PARENT_DIR"/*/; do
  # Remove trailing slash
  dir="${dir%/}"

  # Check if it's actually a directory (handles case where no subdirs exist)
  if [ -d "$dir" ]; then
    echo "Processing directory: $dir"

    uv run impulse/main.py --input $dir --accession_number $(basename "$dir")
    # Replace this with your actual command
    # Example: ls -la "$dir"
    # Example: cd "$dir" && git pull
    # Example: echo "Found: $(basename "$dir")"

    # Your command here:
    echo "Running command on: $(basename "$dir")"
  fi
done

echo "Done processing all subdirectories"
