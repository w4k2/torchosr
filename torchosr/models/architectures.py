from torch import nn
    
def fc_lower_stack(depth, img_size_x, n_out_channels):
    """
    Returns simple architecture with single Fully Connected layer and Relu activatation. 

    :type depth: int
    :param depth: Number of color channels
    
    :type img_size_x: int
    :param img_size_x: Size of image in single axis
    
    :type n_out_channels: int
    :param n_out_channels: Size output

    :rtype: torch.nn.Sequential
    :returns: Lower stack sequential architecture
    """
    
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(depth*img_size_x*img_size_x, n_out_channels),
        nn.ReLU(),
    )
    
def osrci_lower_stack(depth, img_size_x, n_out_channels):
    """
    Returns architecture based on research in OSRCI method.

    :type depth: int
    :param depth: Number of color channels
    
    :type img_size_x: int
    :param img_size_x: Size of image in single axis
    
    :type n_out_channels: int
    :param n_out_channels: Size output

    :rtype: torch.nn.Sequential
    :returns: Lower stack sequential architecture
    """
    
    return nn.Sequential(
    nn.Conv2d(depth, 10, kernel_size=5),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Conv2d(10, 20, kernel_size=5),
    nn.Dropout2d(),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(_osrci_out_size(img_size_x), n_out_channels),
    nn.ReLU()
    )
    
def _osrci_out_size(x):
    x = x-4
    x = int(x/2)
    x = x-4
    x = int(x/2)
    x = x*x
    return 20*x

    
def alexNet32_lower_stack(n_out_channels): 
    """
    Returns modified architecture based on AlexNet, suitable for size (32 x 32 x 3) images.

    :type n_out_channels: int
    :param n_out_channels: Size output

    :rtype: torch.nn.Sequential
    :returns: Lower stack sequential architecture
    """
    return nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3), #30 x 30 x 96
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 15 x 15 x 96
            
            nn.Conv2d(96, 256, kernel_size=3), # 13 x 13 x 256
            nn.ReLU(),
                        
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # 13 x 13 x 384
            nn.ReLU(),
            
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # 13 x 13 x 384
            nn.ReLU(),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 13 x 13 x 256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 6 x 6 x 256
            
            nn.Dropout(),

            nn.Flatten(),
            
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            
            nn.Dropout(),
            
            nn.Linear(4096, n_out_channels),
            nn.ReLU(),
            )