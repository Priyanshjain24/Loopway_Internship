import os
import numpy as np
from utils import params, DataGenerator, jacard_coef, load_test_masks, evaluate
from keras.models import load_model

# Set CUDA_VISIBLE_DEVICES to '-1' to disable GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define the directory containing the data
data_dir = 'data'

# Define the file path for the trained model
model_path = 'models/comma_U_mid.h5'

# Define the directory containing the weights
weights_path = 'weights'

# Define the target shape for input images
target_shape = (480, 288)

# Define the batch size for testing
batch_size = 10

# Create a data generator for testing data
test = DataGenerator(data_dir=data_dir, shape=target_shape, batch_size=batch_size, split='test')

# Load the loss function and metrics from the weights file
total_loss, metrics = params(weights_path)

# Load the trained model
model = load_model(model_path, custom_objects={'dice_loss_plus_1focal_loss': total_loss, 'jacard_coef': jacard_coef})

# Perform predictions on the test data
pred = model.predict(test)

# Load the ground truth masks for the test data
truth_argmax = load_test_masks(data_dir, target_shape)

# Convert the predicted masks to class labels
pred = np.argmax(pred, axis=-1)

# Evaluate the predictions against the ground truth masks
evaluate(truth_argmax, pred)

