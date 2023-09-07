## Training Semantic Segmentation Model 

Clone this repository in your working directory and go in the cloned repo.
All files were developed and tested on python 3.10.6. Install the dependable libraries using:

<pre>
pip install -r requirements.txt
</pre>

Download the "100K Images" and "Drivable Area" dataset from their [official website](https://www.vis.xyz/bdd100k/) and unzip them.

After unzipping is done, you can prepare the complete dataset (100K) by running the shell script as:
<pre>
./prepare.sh 
</pre>

If you want to prepare a smaller dataset (10K) then run the shell script:
<pre>
./small.sh 
</pre>

You will see a new folder named "data" has been created and the original folder has been removed. A new file named 'weights.txt' will also be formed.

To train the model run train.py.
See and run cells of detect.ipynb to detect on new images.

If you ever encounter the error "Killed" while running the scripts, it means you have ran out of RAM Memory.

## Folder Descriptions:
- models: Contains trained models on Squeeze-Net and U-Net.
- torch: Contins files for PyTorch implementation of Semantic Segmentation Model. 

## File Descriptions:
- labels.txt: Defines label names and order for semantic segmentation model training.
- squeezenet.py: Defines the SqueezeSegNet model for semantic segmentation using Keras.
- unet.py: Defines the U-Net model for semantic segmentation using Keras.
- train.py: Trains a semantic segmentation model based on the selected architecture ('U' or 'Squeeze').
- weights.py: Calculates and saves the average class distribution for a semantic segmentation dataset based on the mask images in the training directory.
- weights.txt: Contains the average class distribution across the training dataset.
- utils_general.py: Contains general utility functions like finding total classes, loading images & loading masks.
- utils_model.py: Contains model utility functions for semantic segmentation tasks, including custom loss functions and accuracy metrics.
- unsegmented19.png: test image from CARLA sim.
- prepare.sh: Prepares data for a semantic segmentation task by reorganising directories.
- small.sh: Prepares a smaller dataset for semantic segmentation by shuffling and splitting validation image files into train, validation, and test sets, moving them to the appropriate folders, and removing directories.
- BDD100K_U_last (models): Trained U-Net Model on the BDD100K dataset after the last epoch.
- BDD100K_U_mid.h5 (models): Trained U-Net Model on the BDD100K dataset with the best validation accuracy (jacard coefficient).