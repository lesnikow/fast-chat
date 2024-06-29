#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <search> <replace>"
  exit 1
fi

# Assign positional arguments to variables
search=$1
replace=$2

# Base directory to start the search
base_dir="./data/mt_bench"

# Find and replace in file contents
find "$base_dir" -type f -name "*.jsonl" -exec sed -i "s/${search}/${replace}/g" {} +

# Find and rename files
find "$base_dir" -type f -name "*${search}*.jsonl" | while read -r file; do
  new_file=$(echo "$file" | sed "s/${search}/${replace}/g")
  mv "$file" "$new_file"
  echo "Renamed $file to $new_file"
done
