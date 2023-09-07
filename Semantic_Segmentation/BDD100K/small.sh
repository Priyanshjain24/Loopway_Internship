# Set the paths
images_folder="bdd100k_images_100k/bdd100k/images/100k/val"
masks_folder="BDD100K/bdd100k_drivable_labels_trainval/bdd100k/labels/drivable/masks/val"

data_folder="data"
train_folder="$data_folder/images/train"
val_folder="$data_folder/images/val"
train_masks_folder="$data_folder/masks/train"
val_masks_folder="$data_folder/masks/val"

# Create the necessary folders
mkdir -p "$train_folder" "$val_folder" "$train_masks_folder" "$val_masks_folder"

# Get a list of image files
image_files=("$images_folder"/*)

# Get the total number of images    
total_images=${#image_files[@]}

# Calculate the split sizes
train_split=$((total_images * 9 / 10))
val_split=$((total_images * 1 / 10))

# Shuffle the image files randomly
shuffled_files=($(shuf -e "${image_files[@]}"))

# Copy the images to train and val folders with same basenames
for ((i=0; i<total_images; i++))
do
    image_file="${shuffled_files[i]}"
    image_basename=$(basename "$image_file")
    mask_basename="${image_basename%.jpg}.png"
    mask_file="$masks_folder/$mask_basename"

    if ((i < train_split)); then
        mv "$image_file" "$train_folder/$image_basename"
        mv "$mask_file" "$train_masks_folder/$mask_basename"
    else
        mv "$image_file" "$val_folder/$image_basename"
        mv "$mask_file" "$val_masks_folder/$mask_basename"
    fi
done

rm -rf "bdd100k_images_100k"
rm -rf "BDD100K"

python3 list.py
