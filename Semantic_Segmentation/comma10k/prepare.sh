#!/bin/bash

# Set the source and destination paths
SOURCE_FOLDER="./comma10k"
DESTINATION_FOLDER="./data"

# Set the percentages for train, validation, and test
TRAIN_PERCENTAGE=85
VAL_PERCENTAGE=10
TEST_PERCENTAGE=5

# Create the destination folders
mkdir -p "$DESTINATION_FOLDER/train/imgs"
mkdir -p "$DESTINATION_FOLDER/train/masks"
mkdir -p "$DESTINATION_FOLDER/val/imgs"
mkdir -p "$DESTINATION_FOLDER/val/masks"
mkdir -p "$DESTINATION_FOLDER/test/imgs"
mkdir -p "$DESTINATION_FOLDER/test/masks"

# Delete all files and folders except "imgs" and "masks"
find "$SOURCE_FOLDER" -mindepth 1 -maxdepth 1 ! -name "imgs" ! -name "masks" -exec rm -rf {} +

echo "Cleaned Cloned Repository"

# Copy and split the files into train, validation, and test folders
src_imgs_folder="$SOURCE_FOLDER/imgs"
src_masks_folder="$SOURCE_FOLDER/masks"

echo "Reading Images..."

# Get the list of image files
img_files=($(find "$src_imgs_folder" -type f))
num_imgs=${#img_files[@]}

# Calculate the number of images for each set
num_train=$(echo "$num_imgs * $TRAIN_PERCENTAGE / 100" | bc | awk '{printf("%d\n", $1 + 0.5)}')
num_val=$(echo "$num_imgs * $VAL_PERCENTAGE / 100" | bc | awk '{printf("%d\n", $1 + 0.5)}')
num_test=$(echo "$num_imgs * $TEST_PERCENTAGE / 100" | bc | awk '{printf("%d\n", $1 + 0.5)}')

echo "Shuffling Images..."

# Shuffle the indices of the image files
shuffled_indices=($(seq 0 $((num_imgs - 1)) | shuf))

echo "Splitting and Moving Images..."

# Copy the files to the respective folders
for ((i = 0; i < num_imgs; i++)); do
    index=${shuffled_indices[$i]}
    img_file="${img_files[$index]}"
    base_name=$(basename "$img_file")
    mask_file="$src_masks_folder/$base_name"

    if ((i < num_train)); then
        dest_folder="$DESTINATION_FOLDER/train/imgs"
        dest_masks_folder="$DESTINATION_FOLDER/train/masks"
        txt_file="$DESTINATION_FOLDER/train/train.txt"
    elif ((i < (num_train + num_val))); then
        dest_folder="$DESTINATION_FOLDER/val/imgs"
        dest_masks_folder="$DESTINATION_FOLDER/val/masks"
        txt_file="$DESTINATION_FOLDER/val/val.txt"
    else
        dest_folder="$DESTINATION_FOLDER/test/imgs"
        dest_masks_folder="$DESTINATION_FOLDER/test/masks"
        txt_file="$DESTINATION_FOLDER/test/test.txt"
    fi

    # Copy the image and mask files to the destination folders
    mv "$img_file" "$dest_folder"
    mv "$mask_file" "$dest_masks_folder"

    # Store the filename in the corresponding text file
    echo "$base_name" >> "$txt_file"
done

echo "Files moved and split successfully"
echo "Removing Empty Files..."

# Remove the empty source directories
rmdir "$SOURCE_FOLDER/imgs"
rmdir "$SOURCE_FOLDER/masks"
rmdir "$SOURCE_FOLDER"

echo "Data Prepared Successfully"