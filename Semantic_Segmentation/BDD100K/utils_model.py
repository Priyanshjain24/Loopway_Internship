import segmentation_models as sm
import keras.backend as K

def jacard_coef(y_true, y_pred):
    """
    Calculate the Jaccard coefficient (IoU) between two sets of masks.
    
    Args:
        y_true (tensorflow.Tensor): True masks.
        y_pred (tensorflow.Tensor): Predicted masks.
    
    Returns:
        float: The Jaccard coefficient.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def params(weights_path):
    """
    Load and configure loss function and evaluation metrics.
    
    Args:
        weights_path (str): Path to the file containing weight information.
    
    Returns:
        tuple: A tuple containing the total loss function and the list of evaluation metrics.
    """
    with open(weights_path, 'r') as file:
        weights = [float(line.strip()) for line in file]

    dice_loss = sm.losses.DiceLoss(class_weights=weights)
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = ['accuracy', jacard_coef]

    return total_loss, metrics

