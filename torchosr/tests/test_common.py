"""Basic tests."""

import numpy as np
import torchosr as osr
import torch
from torchosr.data.base_datasets import SVHN_base
from torchosr.data import DataWrapper
from torchosr.utils.base import grayscale_transform
from torchvision import transforms
from torch.utils.data import DataLoader

def get_data(known):
    t_svhn = transforms.Compose([
        grayscale_transform(),
        transforms.Resize(28),
        transforms.ToTensor()])

    base_data = SVHN_base(root='data',transform=t_svhn, download=True)
    train_data = DataWrapper(root='data', 
                    base_dataset=base_data,
                    indexes=np.arange(0,6000),
                    get_classes=np.arange(10),
                    known_classes=known,
                    return_only_known=True,
                    onehot=True,
                    onehot_num_classes=len(known)+1)

    test_data = DataWrapper(root='data', 
                    base_dataset=base_data,
                    indexes=np.arange(6000,7000),
                    get_classes=np.arange(10),
                    known_classes=known,
                    return_only_known=False,
                    onehot=True,
                    onehot_num_classes=len(known)+1)
    
    return train_data, test_data

def test_NoiseSoftmax():
    # Define parameters
    learning_rate = 1e-3
    batch_size = 128
    epochs = 2

    # Select data
    known = [0,1]
    train_data, test_data = get_data(known)
    
    # Prepare data loaders
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model_openmax = osr.models.NoiseSoftmax(n_known=len(known))
    
    # Initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer_openmax = torch.optim.SGD(model_openmax.parameters(), lr=learning_rate)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        inner_score, outer_score, hp_score, overall_score = model_openmax.test(test_data_loader, loss_fn)
        model_openmax.train(train_data_loader, loss_fn, optimizer_openmax)
        
def test_OverlaySoftmax():
    # Define parameters
    learning_rate = 1e-3
    batch_size = 128
    epochs = 2

    # Select data
    known = [0,1]
    train_data, test_data = get_data(known)
    
    # Prepare data loaders
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model_openmax = osr.models.OverlaySoftmax(n_known=len(known))
    
    # Initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer_openmax = torch.optim.SGD(model_openmax.parameters(), lr=learning_rate)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        inner_score, outer_score, hp_score, overall_score = model_openmax.test(test_data_loader, loss_fn)
        model_openmax.train(train_data_loader, loss_fn, optimizer_openmax)
    
def test_Openmax():
    # Define parameters
    learning_rate = 1e-3
    batch_size = 128
    epochs = 2
    epsilon = .5

    # Select data
    known = [0,1]
    train_data, test_data = get_data(known)
    
    # Prepare data loaders
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model_openmax = osr.models.Openmax(n_known=len(known), epsilon=epsilon)
    
    # Initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer_openmax = torch.optim.SGD(model_openmax.parameters(), lr=learning_rate)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        inner_score, outer_score, hp_score, overall_score = model_openmax.test(test_data_loader, loss_fn)
        model_openmax.train(train_data_loader, loss_fn, optimizer_openmax)

def test_Softmax():
    # Define parameters
    learning_rate = 1e-3
    batch_size = 128
    epochs = 2
    epsilon = .5

    # Select data
    known = [0,1]
    train_data, test_data = get_data(known)
    
    # Prepare data loaders
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model_openmax = osr.models.TSoftmax(n_known=len(known), epsilon=epsilon)
    
    # Initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer_openmax = torch.optim.SGD(model_openmax.parameters(), lr=learning_rate)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        inner_score, outer_score, hp_score, overall_score = model_openmax.test(test_data_loader, loss_fn)
        model_openmax.train(train_data_loader, loss_fn, optimizer_openmax)