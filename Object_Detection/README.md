## Custom Object Detection using YOLOv5

All files were developed and tested on python 3.10.6.
Clone this repository in your working directory and go in the cloned repo.

Download Udacity Self Driving Car Dataset from this [link](All files were developed and tested on python 3.10.6.) in your current working directory.
Unzip the folder and rename it to "Udacity_Dataset".
After cloning is done, you can prepare the dataset by running the shell script as:

<pre>
./prepare.sh 
</pre>

Now the contents of the folder "Udacity_dataset" have been modified for Car and Traffic Light detection.
Clone the YOLOv5 implementation by ultralytics in your current directory. You can do this by:

<pre>
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
</pre>

**NOTE:** Make sure that you install correct version of CUDA and cuDNN compatible with your torch version.

To train the model run:
<pre>
python3 train.py --weights yolov5s.pt --data ../Udacity_Dataset/data.yaml --epochs 100 --batch-size 16 --img 512 --device 0
</pre>

You can play around with many parameters (check parser arguments in python files).

## Folder Descriptions:
- trained: Contains trained models on this custom dataset to detect Cars and traffic lights. 

## File Descriptions:
- prepare.sh: Prepares data for Car and Traffic Light Detection by shuffling and splitting image files into train, validation, and test sets, moving them to the appropriate folders, removing unnecessary files and modifying labels of all images for required custom dataset.
- modify.py: Python file to modify class labels for all images to obtain required Custom Dataset of Cars and Traffic Lights.
