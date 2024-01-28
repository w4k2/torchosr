import torch
from torch import nn
from torchmetrics import ConfusionMatrix, Accuracy
import itertools
from .base import OSRModule
import numpy as np
from numpy.matlib import repmat

class OverlaySoftmax(OSRModule):
    """
    Implementation of OverlaySoftmax method

    :type lower_stack: nn.Sequential
    :param lower_stack: Network architecture of lower_stack
    
    :type n_known: int
    :param n_known: Number of known classes
    
    """
    def __init__(self, 
                 lower_stack=None, 
                 n_known=3, 
                 threshold=None, 
                 size=0.5,
                 shuffle=False,
                 n_mixed=2):
        super(OverlaySoftmax, self).__init__(lower_stack=lower_stack,
                                      n_known=n_known)
        self.randy = None
        self.size = size
        self.threshold = threshold
        self.shuffle = shuffle
        self.n_mixed = n_mixed
        
        self.upper_stack = nn.Sequential(
            nn.Linear(64, self.n_known+1)
        )
        
    def train(self, dataloader, loss_fn, optimizer):
        for batch, (X, y) in enumerate(dataloader):
            # Establish number of target samples            
            n_target_samples = int(np.ceil(X.shape[0] * self.size))
            
            # Calculate known class distribution and list present classes
            distribution = y.sum(0).detach().numpy().astype(int)
            present = np.where(distribution != 0)[0]
            
            # Establish all combinations and enhance it to the target number of samples
            pairing = np.array(list(itertools.combinations(present, self.n_mixed)))         
            n_review = len(pairing)
            
            # Secure on single-class batches
            if n_review > 0:
                n_cycles = np.ceil(n_target_samples / n_review).astype(int)
                pairing = repmat(pairing, n_cycles, 1)[:n_target_samples]
                
                # Shuffle pairings if necessary
                if self.shuffle:
                    np.random.shuffle(pairing)
                
                # Prepare storage
                overlayed_X = []
                
                # Iterate pairing
                for pid, elements in enumerate(pairing):
                    # Select sample sublocation
                    locs = pid % distribution[elements]
                    
                    # Establish sample location
                    ids = [np.where(y[:,element] == 1)[0][loc] for (element, loc) in zip(elements, locs)]
                    
                    # Integrate and store sample
                    x = X[ids].mean(0)                
                    overlayed_X.append(x)
                    
                overlayed_X = torch.stack(overlayed_X)            
                overlayed_y = torch.zeros((overlayed_X.shape[0], self.n_known+1))
                overlayed_y[:,-1] = 1
                
                _X = torch.cat((X, overlayed_X))
                _y = torch.cat((y, overlayed_y))
            else:
                _X = X
                _y = y
            
            # Compute prediction and loss
            pred = self(_X)
            loss = loss_fn(pred, _y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
    def test(self, dataloader, loss_fn, conf=False):
        inner_preds = []
        inner_preds_hp = []
        inner_preds_overall = []
        outer_preds = []
        
        inner_targets = []
        outer_targets = []
        overall_targets = []
        
        # Define metric for open set detection
        try:
            inner_metric = Accuracy(task="multiclass", num_classes=self.n_known, average='macro')
            overall_metric = Accuracy(task="multiclass", num_classes=self.n_known+1, average='macro')
        except:
            inner_metric = None
            
        outer_metric = Accuracy(task='multiclass', num_classes=2, average='macro')

        with torch.no_grad():
            for X, y in dataloader:
                # Get y flat and known mask
                y_flat = y.argmax(1)
                known_mask = ~(y_flat == self.n_known)
                
                # Calculate logits for full batch
                logits = self(X)

                # # Establish inner preds and inner y
                inner_pp = nn.Softmax(dim=1)(logits[:,:-1])
                inner_pred = inner_pp.argmax(1)[known_mask]
                inner_target = y_flat[known_mask]
                
                inner_pp_overall = nn.Softmax(dim=1)(logits)
                overall_pred = inner_pp_overall.argmax(1)

                # Establish outer pred
                outer_pred = (logits.argmax(1) != self.n_known).int()

                if self.threshold is not None:
                    low_support_mask = inner_pp_overall.max(1)[0] < self.threshold
                    
                    overall_pred[low_support_mask] = self.n_known                    
                    outer_pred[low_support_mask] = 0
                
                inner_pred_hp = overall_pred[known_mask]

                outer_target = (y_flat != self.n_known).int()
                # print(torch.unique(outer_pred, return_counts=True))
                
                # Store predictions
                inner_preds.append(inner_pred)
                inner_preds_hp.append(inner_pred_hp)
                inner_preds_overall.append(overall_pred)

                inner_targets.append(inner_target)
                overall_targets.append(y_flat)
                
                outer_preds.append(outer_pred)
                outer_targets.append(outer_target)
                
        inner_targets = torch.cat(inner_targets)
        overall_targets = torch.cat(overall_targets)
        inner_preds = torch.cat(inner_preds)
        inner_preds_hp = torch.cat(inner_preds_hp)
        inner_preds_overall = torch.cat(inner_preds_overall)
        
        outer_targets = torch.cat(outer_targets)
        outer_preds = torch.cat(outer_preds)
        
        # Calculate scores
        if inner_metric is not None:
            inner_score = inner_metric(inner_preds, inner_targets)
            inner_score_hp = overall_metric(inner_preds_hp, inner_targets)
            inner_score_overall = overall_metric(inner_preds_overall, overall_targets)
        else:
            inner_score = torch.nan
            inner_score_hp = torch.nan
            inner_score_overall = torch.nan
            
        outer_score = outer_metric(outer_preds, outer_targets)
        
        if conf==True:
            c_inner = ConfusionMatrix(task="multiclass", num_classes=self.n_known)(inner_preds, inner_targets)
            c_outer = ConfusionMatrix(task="binary")(outer_preds, outer_targets)
            c_hp = ConfusionMatrix(task="multiclass", num_classes=self.n_known+1)(inner_preds_hp, inner_targets)
            c_overall = ConfusionMatrix(task="multiclass", num_classes=self.n_known+1)(inner_preds_overall, overall_targets)
            return inner_score, outer_score, inner_score_hp, inner_score_overall, c_inner, c_outer, c_hp, c_overall
            
        if self.verbose:   
            print('Inner metric : %.3f' % inner_score)
            print('Outer metric : %.3f' % outer_score)
            
        return inner_score, outer_score, inner_score_hp, inner_score_overall


    def predict(self, X):
        with torch.no_grad():
            logits = self(X)
            
            overall_pred = nn.Softmax(dim=1)(logits).argmax(1)
            outer_pred = (nn.Softmax(dim=1)(logits).max(1).values > self.epsilon).int()
            overall_pred[outer_pred==0]=self.n_known
            
            return overall_pred
                
        