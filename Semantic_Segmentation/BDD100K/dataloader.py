import os
import numpy as np
from keras.utils import to_categorical, Sequence
from utils_general import load_image, load_mask

class DataGenerator(Sequence):

    def __init__(self, dir, shape, split, num_classes, shuffle=False, batch_size=10):
        """
        DataGenerator class for generating batches of images and masks for training or validation.
        
        Args:
            dir (str): The directory path where the images and masks are stored.
            shape (tuple): The desired shape of the images and masks.
            split (str): The dataset split (e.g., 'train', 'val', 'test').
            num_classes (int): The number of classes or categories.
            shuffle (bool): Whether to shuffle the data at the end of each epoch. Defaults to False.
            batch_size (int): The batch size. Defaults to 10.
        """
        self.dir = dir
        self.shape = shape
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.split = split

        # Load the list of file names from the split file
        with open(os.path.join(dir, 'lists', split+'.txt' ), 'r') as file:
            self.list = [line.strip() for line in file.readlines()]

        # Create an array of indices for shuffling
        self.indices = np.arange(len(self.list))

        # Shuffle the indices at the end of each epoch
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Shuffles the indices at the end of each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """
        Returns the number of batches in the Sequence.
        """
        return int(len(self.list) / self.batch_size)

    def __getitem__(self, index):
        """
        Generates one batch of data.
        
        Args:
            index (int): The index of the batch.
        
        Returns:
            tuple: A tuple containing the batch of images and masks.
        """
        # Get the indices of the current batch
        indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]

        # Initialize empty lists for images and masks
        images = []
        masks = []

        # Load images and masks for the current batch
        for i in indices:
            image_path = os.path.join(self.dir, 'images', self.split, self.list[i].replace(".png", ".jpg"))
            mask_path = os.path.join(self.dir, 'masks', self.split, self.list[i])
            images.append(load_image(image_path, self.shape))
            masks.append(load_mask(mask_path, self.shape))

        # Normalize images and convert masks to one-hot encoding
        images = np.array(images) / 255
        masks = np.expand_dims(masks, axis=3)
        masks = to_categorical(masks, num_classes=self.num_classes)

        return images, masks

