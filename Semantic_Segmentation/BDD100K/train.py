import os
from keras.callbacks import ModelCheckpoint
from squeezenet import squeeze_segNet
from unet import multi_unet_model
from utils_model import params
from utils_general import total_classes
from dataloader import DataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Directory paths and configurations
data_dir = 'data'  # Directory path where the data is stored
input_shape = (480, 288, 3)  # Shape of the input images
batch_size = 16  # Batch size for training
num_epochs = 40  # Number of epochs for training
shuffle_data = True  # Whether to shuffle the data during training
saved_weight_dir = 'models'  # Directory path where the trained weights will be saved
model_name = 'U'  # Model name: 'U' for U-Net, 'Squeeze' for SqueezeSegNet
weights_path = 'weights.txt'  # Path to the file containing weights information

# Define the target shape for the data generator
target_shape = (input_shape[0], input_shape[1])

# Create data generators for training and validation data
num_classes = total_classes(weights_path)  # Get the number of classes from weights information
train_data_generator = DataGenerator(data_dir, target_shape, split='train', num_classes=num_classes,
                                     batch_size=batch_size, shuffle=shuffle_data)
val_data_generator = DataGenerator(data_dir, target_shape, split='val', num_classes=num_classes,
                                   batch_size=batch_size, shuffle=shuffle_data)

if not os.path.exists(saved_weight):
    os.makedirs(saved_weight)

# Create the model
if model_name == 'Squeeze':
    # Create and initialize the SqueezeSegNet model
    context_Net = squeeze_segNet(n_labels=num_classes, image_shape=input_shape)
    model = context_Net.init_model()
elif model_name == 'U':
    # Create the U-Net model
    model = multi_unet_model(num_classes, IMG_HEIGHT=input_shape[1], IMG_WIDTH=input_shape[0],
                             IMG_CHANNELS=input_shape[2])
else:
    print("Wrong Model Name")

# Define paths for saving checkpoints and model weights
mid_model_path = os.path.join(saved_weight_dir, 'BDD100K_' + model_name + '_mid.h5')
fin_model_path = os.path.join(saved_weight_dir, 'BDD100K_' + model_name + '_last.hdf5')

# Load loss function and metrics from weights information
total_loss, metrics = params(weights_path)

# Define a callback to save the best model checkpoint based on validation loss
chk = ModelCheckpoint(mid_model_path, monitor='val_loss', verbose=0, save_best_only=True,
                      save_weights_only=False, mode='auto', save_freq="epoch")

# Compile the model
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)

# Print model summary
model.summary()

# Train the model
model.fit(train_data_generator, epochs=num_epochs, verbose=1, callbacks=[chk], validation_data=val_data_generator)

# Save the final model weights
model.save(fin_model_path)

