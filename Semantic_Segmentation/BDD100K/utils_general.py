import cv2

def total_classes(path):
    """
    Get the total number of classes from a file.
    
    Args:
        path (str): Path to the file containing class information.
    
    Returns:
        int: The total number of classes.
    """
    with open(path, 'r') as file:
        num_classes = len(file.readlines())
        return num_classes

def load_image(path, shape):
    """
    Load and preprocess an image from a given path.
    
    Args:
        path (str): Path to the image file.
        shape (tuple): Desired shape of the image.
    
    Returns:
        numpy.ndarray: The loaded and preprocessed image.
    """
    image = cv2.imread(path)  # Load the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the color channels to RGB
    image = cv2.resize(image, shape)  # Resize the image to the desired shape
    return image

def load_mask(path, shape):
    """
    Load and preprocess a mask from a given path.
    
    Args:
        path (str): Path to the mask file.
        shape (tuple): Desired shape of the mask.
    
    Returns:
        numpy.ndarray: The loaded and preprocessed mask.
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load the mask as grayscale
    mask = cv2.resize(mask, shape)  # Resize the mask to the desired shape
    return mask

