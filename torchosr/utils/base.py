import torch
from PIL import ImageOps
from torchvision.transforms import Lambda

def onehot_bg(n_classes):
    """
    One-hot encoder for classification labels
    """
    return Lambda(lambda y: torch.zeros(n_classes, dtype=torch.float).scatter_(0, y.clone().detach(), value=1))

def inverse_transform():
    "Inverse image values"
    return Lambda(lambda img: 1-img)

def grayscale_transform():
    "Return mean on color channels"    
    return Lambda(lambda img: ImageOps.grayscale(img))
