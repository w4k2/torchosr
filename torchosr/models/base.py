import torch
from torch import nn
from ..architectures.architectures import fc_lower_stack

class OSRModule(nn.Module):
    def __init__(self, n_known, lower_stack=None, verbose=False):
        super(OSRModule, self).__init__()
        self.n_known = n_known
        self.verbose = verbose
        
        if lower_stack is None:
            self.lower_stack = fc_lower_stack(1, 28, 64)
        else:
            self.lower_stack = lower_stack
        
    def forward(self, X):
        representation = self.lower_stack(X)

        logits = self.upper_stack(representation)
        logits = self._norm(logits)
              
        return logits
    
    def train(self, dataloader, loss_fn, optimizer):
        size = len(dataloader.dataset)
        
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self(X)
            loss = loss_fn(pred, y[:,:-1])
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if self.verbose:
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")                  

        
    def _norm(self, x):
        norm = torch.norm(x, p=2, dim=1)
        x = x / (norm.expand(1, -1).t() + .0001)
        return x