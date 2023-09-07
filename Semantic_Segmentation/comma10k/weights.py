import os, cv2
import numpy as np
from labels import label_defs

data_dir = 'data'
batch = 10

# Read the paths of mask images from the training directory
with open(os.path.join(data_dir, 'train/train.txt'), 'r') as file:
    lines = [os.path.join(data_dir, 'train/masks', line.strip()) for line in file.readlines()]

avg = []

# Process images in batches
for i in range(int(len(lines)/batch)):

    temp = np.arange(start=i*10, stop=min((i+1)*10, len(lines)), step=1)
    # Load and reshape images
    images = np.array([cv2.cvtColor(cv2.imread(lines[index]), cv2.COLOR_BGR2RGB) for index in temp]).reshape(-1, 3)

    cat = np.zeros(images.shape[0], dtype=np.uint8)
    # Convert RGB images to categorical labels
    for i, region in enumerate(label_defs):
        cat[np.all(images == region[1], axis=-1)] = i

    # Calculate the class distribution for the batch
    avg.append([float(np.bincount(cat)[i]/cat.shape) for i in range(len(label_defs))])

# Calculate the average class distribution across all batches
avg = np.mean(avg, axis=0)

# Save the average class distribution to a weights file
with open('weights.txt', 'w') as file:
    for i, label in enumerate(label_defs):
        file.write(label[0] + ' : ' + str(avg[i]) + '\n')

