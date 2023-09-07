import cv2
import numpy as np
from utils_general import total_classes

labels = 'labels.txt'  # Path to the file containing class labels
file_list = 'data/lists/train.txt'  # Path to the file containing the list of training files
batch = 10  # Batch size for calculating class weights

# Get the total number of classes from the labels file
num_classes = total_classes(labels)

avg = []  # List to store class weights

with open(file_list, 'r') as file:
    lines = ['data/masks/train/' + line.strip() for line in file.readlines()]

# Iterate over the lines in batches
for i in range(int(len(lines) / batch)):
    temp = np.arange(start=i * 10, stop=min((i + 1) * 10, len(lines)), step=1)
    
    # Load and flatten images from the selected batch
    images = np.array([cv2.imread(lines[index], cv2.IMREAD_GRAYSCALE) for index in temp]).flatten()
    
    # Calculate class weights for the batch and append to the 'avg' list
    avg.append([np.count_nonzero(images == i) / len(images) for i in range(num_classes)])

# Calculate the average class weights across all batches
avg = np.mean(avg, axis=0)

# Save the class weights to the 'weights.txt' file
with open('weights.txt', 'w') as file:
    for i in avg:
        file.write(str(i) + '\n')
