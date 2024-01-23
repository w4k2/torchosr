import torch
from torch import nn
from torchmetrics import Accuracy, ConfusionMatrix
from .base import OSRModule

class GSL(OSRModule):
    def __init__(self, lower_stack=None, n_known=3, sigma=3, n_generated=1.0, verbose=False, normal=False, threshold=None):
        super(GSL, self).__init__(lower_stack=lower_stack,
                                     n_known=n_known)
        self.sigma = sigma
        self.n_generated = n_generated
        self.randy = None
        self.verbose = verbose
        self.normal = normal
        self.threshold = threshold
        
        self.upper_stack = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, self.n_known + 1)
        )
        
        
    def forward(self, X):
        representation = self.lower_stack(X)

        # Input background tension if representation requires grad        
        if representation.requires_grad:
            # Calculate representation
            mean = representation.mean(0)
            std = representation.std(0)
            if self.normal:
                rand_representation = torch.randn((int(self.n_generated*representation.shape[0]), representation.shape[1])) * std * self.sigma + mean
            else:
                rand_representation = torch.rand((int(self.n_generated*representation.shape[0]), representation.shape[1])) * std * self.sigma + mean
            
            # Join representations
            cat_represenation = torch.cat((representation, rand_representation))
            
            # Establish logits on joined
            logits = self.upper_stack(cat_represenation)
            
            # Prepare Randy
            self.randy = torch.zeros((rand_representation.shape[0], logits.shape[1]))
            self.randy[:,-1] = 1
        else:
            logits = self.upper_stack(representation)

        return logits
        
        
    def train(self, dataloader, loss_fn, optimizer, rand_kickstart=False):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            
            if rand_kickstart:
                e_X = torch.rand_like(X)
                e_y = 1-y
                
                X = torch.cat([X,e_X])
                y = torch.cat([y,e_y])
                
            # Compute prediction and loss
            pred = self(X)        
            ry = torch.cat((y, self.randy))        
            loss = loss_fn(pred, ry)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                
    def test(self, dataloader, loss_fn, conf=False): 
        inner_preds = []
        inner_preds_harry_potter = []
        inner_preds_overall = []
        outer_preds = []
        
        inner_targets = []
        outer_targets = []
        overall_targets = []
        
        # Define metrics for open set detection
        try:
            inner_metric = Accuracy(task="multiclass", num_classes=self.n_known, average='macro')
            overall_metric = Accuracy(task="multiclass", num_classes=self.n_known+1, average='macro')
        except:
            inner_metric = None
            overall_metric = None
            
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
                inner_pred_overall = inner_pp_overall.argmax(1)

                # Establish outer pred
                outer_pred = (logits.argmax(1) != self.n_known).int()

                if self.threshold is not None:
                    low_support_mask = inner_pp_overall.max(1)[0] < self.threshold
                    
                    inner_pred_overall[low_support_mask] = self.n_known                    
                    outer_pred[low_support_mask] = 0
                
                inner_pred_harry_potter = inner_pred_overall[known_mask]

                outer_target = (y_flat != self.n_known).int()
 
                # Store predictions
                inner_preds.append(inner_pred)
                inner_preds_harry_potter.append(inner_pred_harry_potter)
                inner_preds_overall.append(inner_pred_overall)
                inner_targets.append(inner_target)
                overall_targets.append(y_flat)
                
                outer_preds.append(outer_pred)
                outer_targets.append(outer_target)
                
        inner_targets = torch.cat(inner_targets)
        overall_targets = torch.cat(overall_targets)
        inner_preds = torch.cat(inner_preds)
        inner_preds_harry_potter = torch.cat(inner_preds_harry_potter)
        inner_preds_overall = torch.cat(inner_preds_overall)
        
        outer_targets = torch.cat(outer_targets)
        outer_preds = torch.cat(outer_preds)
        
        # Calculate scores
        if inner_metric is not None:
            inner_score = inner_metric(inner_preds, inner_targets)
            inner_score_harry = overall_metric(inner_preds_harry_potter, inner_targets)
            inner_score_overall = overall_metric(inner_preds_overall, overall_targets)
        else:
            inner_score = torch.nan
            inner_score_harry = torch.nan
            inner_score_overall = torch.nan
            
        outer_score = outer_metric(outer_preds, outer_targets)

        if conf==True:
            c_inner = ConfusionMatrix(task="multiclass", num_classes=self.n_known)(inner_preds, inner_targets)
            c_outer = ConfusionMatrix(task="binary")(outer_preds, outer_targets)
            c_hp = ConfusionMatrix(task="multiclass", num_classes=self.n_known+1)(inner_preds_harry_potter, inner_targets)
            c_overall = ConfusionMatrix(task="multiclass", num_classes=self.n_known+1)(inner_preds_overall, overall_targets)
            return inner_score, outer_score, inner_score_harry, inner_score_overall, c_inner, c_outer, c_hp, c_overall
           
        if self.verbose:
            print('Inner metric : %.3f' % inner_score)
            print('Outer metric : %.3f' % outer_score)
            
        return inner_score, outer_score, inner_score_harry, inner_score_overall


    def predict(self, X): 
        with torch.no_grad():

            logits = self(X)
            pred_overall = nn.Softmax(dim=1)(logits).argmax(1)

            return pred_overall
            