import torch
from PIL import ImageOps
from torchvision.transforms import Lambda

def onehot_bg(n_classes):
    """
    One-hot encoder for classification labels
    """
    return Lambda(lambda y: torch.zeros(n_classes, dtype=torch.float).scatter_(0, y.clone().detach(), value=1))

def inverse_transform():
    """Inverse image values"""
    return Lambda(lambda img: 1-img)

def grayscale_transform():
    """Return mean on color channels"""    
    return Lambda(lambda img: ImageOps.grayscale(img))

def get_openmax_epsilon(n_known):
    """Optimized epsilon value for Openmax and given number of KKC"""
    return {9:.1, 8:.1, 7:.1, 6:.1,
            5:.2, 4:.4, 3:.5, 2:.6}[n_known]
        
def get_softmax_epsilon(n_known):
    """Optimized epsilon value for Thresholded Softmax and given number of KKC"""
    return {10: .04, 9:.05, 8:.1, 7:.2, 6:.25, 
            5:.35, 4:.48, 3:.6, 2:.8}[n_known]