import os
from keras.callbacks import ModelCheckpoint
import squeezenet as sn
from unet import multi_unet_model
from utils import params, DataGenerator
from labels import num_classes

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

# Define the model name ('U' or 'Squeeze')
model_name = 'U'

# Define the file path to save the weights
weights_path = 'weights.txt'

# Define the target shape for input images
target_shape = (input_shape[0], input_shape[1])

# Get the number of classes from the label definitions
n_labels = num_classes()

# Create a data generator for training data
train = DataGenerator(data_dir=data_dir, shape=target_shape, split='train', batch_size=batch_size, shuffle=shuffle)

# Create a data generator for validation data
val = DataGenerator(data_dir=data_dir, shape=target_shape, split='val', batch_size=batch_size, shuffle=shuffle)

# Create the directory to save the trained weights if it doesn't exist
if not os.path.exists(saved_weight):
    os.makedirs(saved_weight)

# Initialize the model based on the selected model name
if model_name == 'Squeeze':
    context_Net = sn.squeeze_segNet(n_labels=n_labels, image_shape=input_shape)
    model = context_Net.init_model()
elif model_name == 'U':
    model = multi_unet_model(n_labels, IMG_HEIGHT=input_shape[1], IMG_WIDTH=input_shape[0], IMG_CHANNELS=input_shape[2])
else:
    print("Wrong Model Name")

# Set the file paths for the intermediate and final trained model
mid_model = os.path.join(saved_weight, 'comma_' + model_name + '_mid.h5')
fin_model = os.path.join(saved_weight, 'comma_' + model_name + '_last.hdf5')

# Load the loss function and metrics from the weights file
total_loss, metrics = params(weights_path)

# Create a ModelCheckpoint callback to save the best weights during training
chk = ModelCheckpoint(mid_model, monitor='val_loss', verbose=0, save_best_only=True,
                      save_weights_only=False, mode='auto', save_freq="epoch")

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)

# Print the summary of the model
model.summary()

# Train the model
model.fit(train, epochs=epochs, verbose=1, callbacks=[chk], validation_data=val)

# Save the final trained model
model.save(fin_model)
