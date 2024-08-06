#!/bin/bash

# Step a: Ask the user to specify an audio input directory
read -p "Please specify the audio input directory: " input_dir

# Remove any extra quotes from the input_dir
input_dir=$(echo $input_dir | tr -d "'")

# Check if the specified directory exists
if [ ! -d "$input_dir" ]; then
  echo "Error: The specified directory does not exist."
  exit 1
fi

# Step b: Wipe all contents from the folder data/
rm -rf data/*

# Step c: Add two new empty folders in data, data/raw and data/wavs
mkdir -p data/raw
mkdir -p data/wavs

# Step d: Run the python script "split_dataset.py"
export INPUT_DIR="$input_dir"
export OUTPUT_DIR="data/wavs"
python split_dataset.py

# Step e: Generate train_files.txt and test_files.txt with full paths
if [ "$(ls -A data/wavs/*.wav 2>/dev/null)" ]; then
  find "$PWD/data/wavs" -name "*.wav" | tail -n+10 > data/train_files.txt
  find "$PWD/data/wavs" -name "*.wav" | head -n10 > data/test_files.txt
else
  echo "No wav files found in data/wavs."
  exit 1
fi

# Step f: Run the training script
env PYTHONPATH="$PWD:$PYTHONPATH" python scripts/train.py --save_path checkpoints --data_path data

