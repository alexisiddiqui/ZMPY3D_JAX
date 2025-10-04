#!/bin/bash

# Exit on any error
set -e

echo "Step 1: Locating ZMPY3D package folder..."
ZMPY3D_PATH=$(python -c "import ZMPY3D; print(ZMPY3D.__file__)")
echo "Found ZMPY3D at: $ZMPY3D_PATH"

# Extract the directory path (remove __init__.py from the end)
ZMPY3D_DIR=$(dirname "$ZMPY3D_PATH")
echo "ZMPY3D directory: $ZMPY3D_DIR"

# Step 2: Navigate to cache_data folder
CACHE_DIR="$ZMPY3D_DIR/cache_data"
echo -e "\nStep 2: Checking cache_data folder..."

if [ ! -d "$CACHE_DIR" ]; then
    echo "Creating cache_data folder at: $CACHE_DIR"
    mkdir -p "$CACHE_DIR"
else
    echo "cache_data folder exists at: $CACHE_DIR"
fi

# Step 3: Download the cache file
echo -e "\nStep 3: Downloading cache file (1.3 GB)..."
cd "$CACHE_DIR"

FILE_ID="1RR1rF_5YJqaxNC5AK0Ie_8MswGb0Tttw"

# Download using gdown (keeps original filename)
gdown "https://drive.google.com/uc?id=$FILE_ID"

echo -e "\nDownload complete!"
echo "File saved to: $CACHE_DIR"
ls -lh "$CACHE_DIR"