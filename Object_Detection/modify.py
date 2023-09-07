import os

dataset = 'Udacity_Dataset'  # Dataset directory

def modify(elements):
    """
    Modify the elements based on the label values.
    
    Args:
        elements (list): List of elements to be modified.
    
    Returns:
        list: Modified list of elements.
    """
    new = []
    
    for i in range(len(elements)):
        label = int(elements[i][0])
        box = elements[i][1:]

        if label == 1:
            new.append('0' + box)
        elif 2 < label < 10:
            new.append('1' + box)
        else:
            pass

    return new

# Traverse the directory tree
for path, folders, files in os.walk(dataset):
    dirname = path.split(os.path.sep)[-1]

    if dirname == 'labels':
        # Process the files in the 'labels' directory
        for txtfile in os.listdir(path):
            dir = os.path.join(path, txtfile)

            with open(dir, 'r') as file:
                elements = file.readlines()
                elements = modify(elements)

            os.remove(dir)

            with open(dir, 'w') as file:
                file.writelines(elements)

