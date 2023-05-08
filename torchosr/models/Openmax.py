import torch
from torch import nn
from torchmetrics import Accuracy, ConfusionMatrix
from .Softmax import Softmax
from scipy.stats import exponweib
import numpy as np

class Openmax(Softmax):
    def __init__(self, lower_stack=None, n_known=3, epsilon=.5, tail=20, alpha=2):
        super(Openmax, self).__init__(lower_stack=lower_stack,
                                      n_known=n_known,
                                      epsilon=epsilon)
        self.tail = tail
        self.alpha = alpha
        self.weibs = {}

    def train(self, dataloader, loss_fn, optimizer):
        size = len(dataloader.dataset)
        
        y_flat = []
        correct = []
        activations = []
        
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self(X)
            loss = loss_fn(pred, y[:,:-1]) # funkcja straty nie bierze udziału, więc ją pożegnajmy
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Estimate correctly classified examples
            _pred_flat = pred.argmax(1)
            _y_flat = y.argmax(1)            
            _correct = torch.eq(_pred_flat, _y_flat)

            activations.append(pred)
            y_flat.append(_y_flat)
            correct.append(_correct)
        
            if self.verbose:
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        activations = torch.cat(activations)
        correct = torch.cat(correct)
        y_flat = torch.cat(y_flat)
        
        activations = activations[correct]
        y_flat = y_flat[correct]        
        
        # Establish mean activation vectors for correctly classified known classes
        # As discussed previously (Alg. 1), we consider the penultimate layer (fully connected layer 8 , i.e., FC8) for computation of mean activation vectors (MAV). The MAV vector is computed for each class by consider- ing the training examples that deep networks training clas- sified correctly for the respective class. MAV is computed for each crop/channel separately.
        
        # TODO: wprowadź parametr alpha i proguj klasy po liczności
        # TODO: dopuszczalna alpha -1, gdzie wpuszcza wszystko o co najmniej tail poprawnie sklasyfikowanych obiektów
        
        n_known_classes = activations.shape[1]
        self.weibs = {}
        
        for label in range(n_known_classes):
            correct_subset = activations[y_flat == label]
            if correct_subset.shape[0] > self.tail:
                mav = correct_subset.mean(0).view(1, -1)

                # Distance between each correctly classified training example and MAV for particu- lar class is computed to obtain class specific distance dis- tribution. For these experiments we use a distance that is a weighted combination of normalized Euclidean and cosine distances.
                # Zastosujmy euklidesowy, skoro sami autorzy stwierdzili, że nie ma różnicy.            
                # Policz tu dystanse pomiędzy correct_subset i mav.            
                dist = torch.cdist(mav, correct_subset)
                
                # Zostaw tylko ogon w dist
                dist = dist.sort()[0][:,-self.tail:]
                
                # Dopasuj rozkład
                proba = dist[0].detach().numpy()
                try:
                    self.weibs.update({label: exponweib.fit(proba, loc=0, scale=.2)})
                except:
                    print('[WEIB] Skipped', proba)
        
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
            inner_metric = Accuracy(task="multiclass", num_classes=self.n_known, average='weighted')
            overall_metric = Accuracy(task="multiclass", num_classes=self.n_known+1, average='weighted')
        except:
            inner_metric = None
            overall_metric = None
        outer_metric = Accuracy(task='binary', average='weighted')

        with torch.no_grad():
            for X, y in dataloader:
                # Get y flat and known mask
                y_flat = y.argmax(1)
                known_mask = ~(y_flat == self.n_known)
                
                # Calculate logits for full batch
                logits = self(X)
                slogit = logits.sum(1)
                
                # Rank activations for alpha trimming
                a_trimmer = 1 - torch.nn.functional.one_hot(torch.topk(logits, self.alpha, dim=1)[1],
                                                            logits.shape[1]).sum(1)
                                
                # Calculate weights
                w = np.ones(logits.shape).astype(float)
                for i in self.weibs:
                    rv = self.weibs[i]
                    w[:,i] -= exponweib.cdf(logits[:,i], *rv)
                
                w = torch.Tensor(w)
                try:
                    w[a_trimmer] = 1
                except:
                    print('[a_trimmer] Skipped', w, a_trimmer)
                
                # Establish weighted logits and unknown activation
                v_logits = w*logits
                plogit = v_logits.sum(1)
                unknown_activation = torch.sub(slogit, plogit)

                # Calculate corrected activation
                logits = torch.cat((v_logits, unknown_activation[:,None]), 1)          

                # Establish inner preds and inner y
                inner_pp = nn.Softmax(dim=1)(logits[:,:-1])
                inner_pred = inner_pp.argmax(1)[known_mask]
                inner_target = y_flat[known_mask]           
                             
                # Establish outer pred
                # Here is softmax
                # Czy wsparcie jest dla klasy znanej
                pred_known = (nn.Softmax(dim=1)(logits).argmax(1) != self.n_known).int() # 1=KKC

                # Czy wsparcie jest dostateczne do zachowania decyzji
                pred_sure = (nn.Softmax(dim=1)(logits).max(1).values > self.epsilon).int() # jeżeli wsparcie jest większe to KKC

                # pred_known = 1-pred_known
                outer_pred = pred_known * pred_sure
                outer_target = (y_flat != self.n_known).int()
                
                inner_pp_overall = nn.Softmax(dim=1)(logits)
                inner_pred_overall = inner_pp_overall.argmax(1)
                inner_pred_overall[outer_pred==0]=self.n_known
                inner_pred_hp= inner_pred_overall[known_mask]
                
                # Store predictions
                inner_preds.append(inner_pred)
                inner_preds_hp.append(inner_pred_hp)
                inner_preds_overall.append(inner_pred_overall)
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
