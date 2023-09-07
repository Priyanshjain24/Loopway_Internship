#!/bin/bash

# Define paths
data_dir="./Udacity_Dataset"

find "$data_dir" -mindepth 1 -maxdepth 1 ! -name "export" -exec rm -r {} +
mv "$data_dir/export" "$data_dir/train"
mv data.yaml "$data_dir/data.yaml"

train_dir="$data_dir/train"
val_dir="$data_dir/val"
test_dir="$data_dir/test"

# Create validation and test directories
mkdir -p "$val_dir/images" "$val_dir/labels"
mkdir -p "$test_dir/images" "$test_dir/labels"

# Copy files from train to val and test directories
image_files=($(ls "$train_dir/images"))
label_files=($(ls "$train_dir/labels"))

# Randomly shuffle the file list
shuf -e "${image_files[@]}" --random-source=/dev/urandom

# Calculate the split sizes
total_files=${#image_files[@]}
train_split=$((total_files * 8 / 10))
val_split=$((total_files * 1 / 10))

echo "Starting Splitting"

# Copy files to validation and test directories
for ((i = 0; i < total_files; i++))
do
  filename=$(basename "${image_files[$i]}")

  if ((i >= train_split && i < train_split + val_split)); then
    mv "$train_dir/images/$filename" "$val_dir/images/"
    mv "$train_dir/labels/${filename%.jpg}.txt" "$val_dir/labels/"
  elif ((i >= train_split + val_split)); then
    mv "$train_dir/images/$filename" "$test_dir/images/"
    mv "$train_dir/labels/${filename%.jpg}.txt" "$test_dir/labels/"
  fi
done 

# Print summary
echo "Splitting completed"
echo "Modifying Classes"

python3 modify.py

echo "Prepared modified dataset"
