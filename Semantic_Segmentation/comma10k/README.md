## Training Semantic Segmentation Model 

Clone this repository in your working directory and go in the cloned repo.

All files were developed and tested on python 3.10.6. Install the dependable libraries using:

<pre>
pip install -r requirements.txt
</pre>

Now, clone this [comma github repo](https://github.com/commaai/comma10k) in your current working directory.

After cloning is done, you can prepare the dataset by running the shell script as:

<pre>
./prepare.sh 
</pre>

You will see a new folder named "data" has been created and the original folder has been removed. 

To train the model, we first need to generate the weights list. To do this, run the weights.py which takes around 20-25 mins. You will see a file called weights.txt once the program has run successfully. 

To train the model run train.py.
To test the accuracy of the model run test.py
To retrain the model run retrain.py

If you ever encounter the error "Killed" while running the scripts, it means you have ran out of RAM Memory.

## Folder Descriptions:
- models: Contains trained models on Squeeze-Net and U-Net,
- torch: Contins files for PyTorch implementation of Semantic Segmentation Model. 

## File Descriptions:
- labels.py: Defines label definitions with names and colors for semantic segmentation model training.
- retrain.py: Re-trains a semantic segmentation model by loading the pre-trained model, and saving the best weights and final trained model after retraining on the improved dataset.
- squeezenet.py: Defines the SqueezeSegNet model for semantic segmentation using Keras.
- test.py: Evaluates the performance of a semantic segmentation model by generating predictions on test data and comparing them to ground truth masks.
- train.py: Trains a semantic segmentation model based on the selected architecture ('U' or 'Squeeze').
- unet.py: Defines the U-Net model for semantic segmentation using Keras.
- utils.py: Contains utility functions for semantic segmentation tasks, including data preprocessing, datapostprocessing, evaluation metrics, data loading, etc.
- weights.py: Calculates and saves the average class distribution for a semantic segmentation dataset based on the mask images in the training directory.
- Squeeze_Net.pdf: Original Research Paper of Squeeze_Net which has been modified.
- U_Net.pdf: Original Research Paper of U_Net which has been modified.
- prepare.sh: Prepares data for a semantic segmentation task by shuffling and splitting image files into train, validation, and test sets, moving them to the appropriate folders, and removing empty directories.
- weights.txt: Contains the average class distribution across the training dataset.
- comma_Squeeze_last.hdf5 (models): Trained Squeeze-Net Model on the comma10k dataset after the last epoch. 
- comma_Squeeze_mid.h5 (models): Trained Squeeze-Net Model on the comma10k dataset with the best validation accuracy (jacard coefficient).
- comma_U_last.hdf5 (models): Trained U-Net Model on the comma10k dataset after the last epoch.
- comma_U_mid.h5 (models): Trained U-Net Model on the comma10k dataset with the best validation accuracy (jacard coefficient).
- unet.py (torch): Defines the U-Net model for semantic segmentation using PyTorch. This was written completely.
- squeezenet.py (torch): Defines the SqueezeSegNet model for semantic segmentation using PyTorch. This is incomplete.
