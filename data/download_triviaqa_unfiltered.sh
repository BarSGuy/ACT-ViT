#!/usr/bin/env bash
set -e

ARCHIVE_URL="https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz"
DATA_DIR="data"
TARGET_DIR="$DATA_DIR/triviaqa-unfiltered"
TMP_DIR="$DATA_DIR/.triviaqa_tmp"

echo "Downloading TriviaQA unfiltered dataset..."

# Create base directories
mkdir -p "$DATA_DIR"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

# Download archive
curl -L "$ARCHIVE_URL" -o "$TMP_DIR/triviaqa-unfiltered.tar.gz"

echo "Extracting archive..."
tar -xzf "$TMP_DIR/triviaqa-unfiltered.tar.gz" -C "$TMP_DIR"

# The archive extracts to triviaqa-unfiltered/
EXTRACTED_DIR="$TMP_DIR/triviaqa-unfiltered"

# Move into final expected location
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

mv "$EXTRACTED_DIR"/unfiltered-web-train.json "$TARGET_DIR/"
mv "$EXTRACTED_DIR"/unfiltered-web-dev.json   "$TARGET_DIR/"

# Cleanup
rm -rf "$TMP_DIR"

echo "Done."
echo "Dataset ready at: $TARGET_DIR"
