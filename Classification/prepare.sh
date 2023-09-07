original_directory="ITSDC-Udacity-Traffic-Light-Classifier"
new_directory="Udacity_Traffic_Light"

# Create a new folder named "Udacity_Traffic_Light"
mkdir "$new_directory"

# Move the "training" and "test" folders into the new folder
mv "$original_directory/traffic_light_images/training" "$new_directory/train"
mv "$original_directory/traffic_light_images/test" "$new_directory/val"

# Delete the original folder "ITSDC-Udacity-Traffic-Light-Classifier"
rm -rf "$original_directory"
