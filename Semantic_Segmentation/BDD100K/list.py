import os

dir = 'data/masks'  # Path to the directory containing masks
list = 'data/lists'  # Path to the directory where the list files will be created

os.mkdir(list)  # Create the 'data/lists' directory if it doesn't exist

for folder in os.listdir(dir):
    # Iterate over each folder in 'data/masks'

    with open(list+'/'+folder+'.txt', 'w') as file:
        # Open a file in write mode for the current folder

        # Write each file name in the current folder to the file, separated by a newline character
        file.writelines(line+'\n' for line in os.listdir(os.path.join(dir, folder)))
