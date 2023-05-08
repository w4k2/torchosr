import torch
from ..data.DataWrapper import DataWrapper
from ..data.OutlierDataset import OutlierDataset
import numpy as np
import copy

def configure_division_outlier(base_dataset, outlier_dataset, repeats, n_openness=None, seed=None, min_known_classes=2):
        
        trng = torch.random.manual_seed(seed)

        # Wczytanie bazowego zbioru, określenie liczby instancji i klas
        n_classes_base = (base_dataset._n_classes())
        n_classes_out = (outlier_dataset._n_classes())

        # Tablice na liczności klas znanych/nieznanych i openness w zależności od nich
        openness_n_classes = []
        openness = []
        
        for i in range(min_known_classes,n_classes_base):
                for j in range(1,n_classes_out):
                      openness_n_classes.append([i,j])  
                      openness.append(1 - np.sqrt((2*i)/((i*2)+j)))

        openness = torch.tensor(openness)
        openness_n_classes = torch.tensor(openness_n_classes)
        
        if n_openness is not None:
                # Losujemy n_openness konfiguracji
                # Faworyzujemy te podzbiory, w których sumaryczna liczba klas jest duża
                p = openness_n_classes.sum(1)
                p = p / p.sum()
                rand_indexes = p.multinomial(num_samples=n_openness, replacement=False, generator=trng)
                
                # Podranie wylosowanych liczności i openness
                op_choice = openness[rand_indexes]
                op_n_classes_choice = openness_n_classes[rand_indexes]
        else:
                # W innym wypadku pobieramy wszystkie
                op_choice = openness
                op_n_classes_choice = openness_n_classes

        #Teraz dla wylosowanych liczności będziemy losować klasy
        repeats_config =[]

        for kkc_n, uuc_n in op_n_classes_choice:    
                #Dla liczby powtórzeń losujemy indeksy kkc i uuc
                for r in range(repeats):
                        all_kkc = torch.arange(n_classes_base, dtype=float)
                        kkc = all_kkc.multinomial(num_samples=kkc_n, replacement=False, generator=trng)
                        
                        all_uuc = torch.arange(n_classes_out, dtype=float)
                        uuc = all_uuc.multinomial(num_samples=uuc_n, replacement=False, generator=trng)
                        
                        #Zapisujemy konfiguracje
                        repeats_config.append((kkc, uuc))
        
        return repeats_config, op_choice


def get_train_test_outlier(base_dataset, outlier_dataset, kkc_indexes, uuc_indexes, root, tunning, fold, seed=1410, n_folds=5):
        trng = torch.random.manual_seed(seed)

        # Teraz można określić prawdziwy zbiór
        data = DataWrapper(root=root,
                       base_dataset=base_dataset,
                       indexes='all',
                       get_classes=kkc_indexes,
                       known_classes=kkc_indexes,
                       return_only_known=True,
                       onehot=True,
                       onehot_num_classes=len(kkc_indexes)+1
                       )
        
        labels = np.unique(data.original_targets)
        
        proportion = .1
        
        # First tuning, later validation
        division_probability = torch.ones(n_folds*2) * (1-proportion) / n_folds
        division_probability[:n_folds] = proportion / n_folds
        
        division_mask = np.zeros_like(data.targets)
        
        for label in labels:
                class_mask = data.original_targets == label

                division_mask[class_mask] = division_probability.multinomial(class_mask.sum(0), replacement=True, generator=trng)                
                
        if tunning:
                test_indexes = division_mask == fold
                train_indexes = division_mask < n_folds
                
                train_indexes = train_indexes * ~test_indexes
        else:
                test_indexes = division_mask == (fold+n_folds)
                train_indexes = division_mask >= n_folds
                
                train_indexes = train_indexes * ~test_indexes

        data_test = copy.deepcopy(data)
        data_train = copy.deepcopy(data)
                
        data_test.reindex(np.where(test_indexes)[0])
        data_train.reindex(np.where(train_indexes)[0])
        
        outliers = DataWrapper(root=root,
                       base_dataset=outlier_dataset,
                       indexes='all',
                       get_classes=uuc_indexes,
                       known_classes=uuc_indexes,
                       return_only_known=True,
                       onehot=False,
                       onehot_num_classes=None
                       )
        
        outliers.targets = [torch.zeros(len(kkc_indexes)+1, dtype=torch.float)
                                         .scatter_(0, torch.tensor(len(kkc_indexes)), value=1)
                                         for t in outliers.targets]
        
        data_test = OutlierDataset(root, data_test, outliers, shuffle=False, onehot=False, unknown_label=len(kkc_indexes))
        
        return data_train, data_test
