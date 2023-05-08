import torch
from PIL import ImageOps
from torchvision.transforms import Lambda

def onehot_bg(n_classes):
    """
    Convert evaluator scores to accumulative mean.

    It's the best way to make reader capable to understand anything from your results.

    :type scores: array-like, shape (n_estimators, n_chunks, n_metrics)
    :param scores: Evaluation scores.

    :rtype: array-like, shape (n_estimators, n_chunks, n_metrics)
    :returns: Evaluation scores in format possible to read for human being.
    """
    return Lambda(lambda y: torch.zeros(n_classes, dtype=torch.float).scatter_(0, y.clone().detach(), value=1))

def inverse_transform():
    "Inverse image values"
    return Lambda(lambda img: 1-img)

def grayscale_transform():
    "Return mean on color channels"    
    return Lambda(lambda img: ImageOps.grayscale(img))
