import os
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from utils import params, DataGenerator, jacard_coef
from labels import num_classes

# Set CUDA_VISIBLE_DEVICES to '-1' to disable GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define the directory containing the data
data_dir = 'data'

# Define the input shape of the model
input_shape = (480, 288, 3)

# Define the batch size for training
batch_size = 10

# Define the number of epochs for training
epochs = 50

# Set shuffle to True to shuffle the data during training
shuffle = True

# Define the directory to save the trained weights
saved_weight = 'models'

# Define the file path to save the weights
weights_path = 'weights.txt'

# Define the file path for the pre-trained model
model_path = 'comma_U_last.hdf5'

# Define the name of the checkpoint file
checkpoint_name = 'check.h5'

# Define the name of the final model file after retraining
final_model = 'retrain.hdf5'

# Define the target shape for input images
target_shape = (input_shape[0], input_shape[1])

# Get the number of classes from the label definitions
n_labels = num_classes()

# Create a data generator for training data
train = DataGenerator(data_dir=data_dir, shape=target_shape, split='train', batch_size=batch_size, shuffle=shuffle)

# Create a data generator for validation data
val = DataGenerator(data_dir=data_dir, shape=target_shape, split='val', batch_size=batch_size, shuffle=shuffle)

# Load the loss function and metrics from the weights file
total_loss, metrics = params(weights_path)

# Load the pre-trained model
model = load_model(model_path, custom_objects={'dice_loss_plus_1focal_loss': total_loss, 'jacard_coef': jacard_coef})

# Create a ModelCheckpoint callback to save the best weights during training
chk = ModelCheckpoint(os.path.join(saved_weight, checkpoint_name), monitor='val_loss', verbose=0,
                      save_best_only=True, save_weights_only=False, mode='auto', save_freq="epoch")

# Train the model
model.fit(train, epochs=epochs, verbose=1, callbacks=[chk], validation_data=val)

# Save the final trained model
model.save(os.path.join(saved_weight, final_model))
