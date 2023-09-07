data_folder="data"

# Create a new folder named "data"
mkdir "$data_folder"

# Copy the "masks" folder into the "data" folder
find BDD100K -type d -name "masks" -exec mv {} "$data_folder" \;

# Copy the "100k" folder into the "data" folder and name it "images"
find bdd100k_images_100k -type d -name "100k" -exec mv {} "$data_folder/images" \;

# Delete the "masks" folder
rm -r BDD100K

# Delete the "images" folder
rm -r bdd100k_images_100k

# Generate file lists
python3 list.py

