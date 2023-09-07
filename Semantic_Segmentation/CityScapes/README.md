## Training Semantic Segmentation Model 

Clone this repository in your working directory and go in the cloned repo.

All files were developed and tested on python 3.10.6. Install the dependable libraries using:

<pre>
pip install -r requirements.txt
</pre>

Download a smaller modified version of the CityScapes from [here](https://iitbacin-my.sharepoint.com/:f:/g/personal/210070063_iitb_ac_in/Et9UUw3JLf1LrTp2_aEL8doBbIZWM_F0aolipszqS9gzxg?e=Md1fsl).

To train the model run train.py.
To test the accuracy of the model run test.py

If you ever encounter the error "Killed" while running the scripts, it means you have ran out of RAM Memory.

## File Descriptions:
- squeezenet.py: Defines the SqueezeSegNet model for semantic segmentation using Keras.
- test.py: Evaluates the performance of a semantic segmentation model by generating predictions on test data and comparing them to ground truth masks.
- train.py: Trains a semantic segmentation model based on the selected architecture ('U' or 'Squeeze').
- unet.py: Defines the U-Net model for semantic segmentation using Keras.
- utils.py: Contains utility functions for semantic segmentation tasks, including data preprocessing, datapostprocessing, evaluation metrics, data loading, etc.
