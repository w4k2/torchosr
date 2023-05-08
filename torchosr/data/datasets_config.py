from ..data.DataWrapper import DataWrapper
import numpy as np
import copy
import torch

def configure_division(base_dataset, repeats, n_openness=None, seed=None, min_known_classes=2):
        # Określenie generatora pseudolosowego
        trng = torch.random.manual_seed(seed)
        
        # Wczytanie bazowego zbioru, określenie liczby instancji i klas
        n_classes = (base_dataset._n_classes())

        # Tablice na liczności klas znanych/nieznanych i openness w zależności od nich        
        kkc, uuc = torch.triu_indices(n_classes-1, n_classes-1)
        kkc += 1
        uuc = n_classes - 1 - uuc
        
        # Remove all the entries that have less known classes than given minimum (default=2)
        kkc, uuc = kkc[kkc>=min_known_classes], uuc[kkc>=min_known_classes]

        openness_n_classes = torch.stack((kkc,uuc)).T        
        openness = 1 - torch.sqrt((2*kkc)/((2*kkc)+uuc))

        # Jeżeli mamy n_openness losujemy n konfiguracji
        if n_openness is not None:
                # Faworyzujemy te podzbiory, w których sumaryczna liczba klas jest duża -- TBD -- (żeby wykorzystać tyle ze zbioru ile się da?)
                p = openness_n_classes.sum(1)
                p = p / p.sum()
                rand_indexes = p.multinomial(num_samples=n_openness, replacement=False, generator=trng)

                # Pobranie wylosowanych liczności i openness
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
                        all_classes = torch.ones(n_classes)
                        kkc = all_classes.multinomial(num_samples=kkc_n, replacement=False, generator=trng)
                        all_classes[kkc] = 0
                        uuc = all_classes.multinomial(num_samples=uuc_n, replacement=False, generator=trng)
                        
                        repeats_config.append((kkc, uuc))
 
        return repeats_config, op_choice


def configure_oneclass_division(base_dataset, repeats, n_openness=None, seed=None):
        # Określenie generatora pseudolosowego
        trng = torch.random.manual_seed(seed)
        
        # Wczytanie bazowego zbioru, określenie liczby instancji i klas
        n_classes = (base_dataset._n_classes())

        # Tablice na liczności klas znanych/nieznanych i openness w zależności od nich
        openness_n_classes = []
        openness = []
        
        for j in range(1,n_classes-1):
                openness_n_classes.append([1,j])
                openness.append(1 - np.sqrt((2)/(2+j)))
                
        openness = torch.tensor(openness)
        openness_n_classes = torch.tensor(openness_n_classes)       
        
        # Jeżeli mamy n_openness losujemy n konfiguracji
        if n_openness is not None:
                # Faworyzujemy te podzbiory, w których sumaryczna liczba klas jest duża -- TBD -- (żeby wykorzystać tyle ze zbioru ile się da?)
                p = openness_n_classes.sum(1)
                p = p / p.sum()
                rand_indexes = p.multinomial(num_samples=n_openness, replacement=False, generator=trng)

                # Pobranie wylosowanych liczności i openness
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
                        all_classes = torch.ones(n_classes)
                        kkc = all_classes.multinomial(num_samples=kkc_n, replacement=False, generator=trng)
                        all_classes[kkc] = 0
                        uuc = all_classes.multinomial(num_samples=uuc_n, replacement=False, generator=trng)
                        
                        repeats_config.append((kkc, uuc))
 
        return repeats_config, op_choice


def get_train_test(base_dataset, kkc_indexes, uuc_indexes, root, tunning, fold, seed=1410, n_folds=5):
        trng = torch.random.manual_seed(seed)
        
        classes = np.concatenate((kkc_indexes, uuc_indexes))

        # Teraz można określić prawdziwy zbiór
        data = DataWrapper(root=root,
                       base_dataset=base_dataset,
                       indexes='all',
                       get_classes=classes,
                       known_classes=kkc_indexes,
                       return_only_known=False,
                       onehot=True,
                       onehot_num_classes=len(kkc_indexes)+1
                       )
        
        labels = data.original_targets.unique()
        
        proportion = .1
        
        # First tuning, later validation
        division_probability = torch.ones(n_folds*2) * (1-proportion) / n_folds
        division_probability[:n_folds] = proportion / n_folds
        
        division_mask = torch.zeros_like(data.targets)
        
        for label in labels:
                class_mask = data.original_targets == label
                division_mask[class_mask] = division_probability.multinomial(class_mask.sum(0), replacement=True, generator=trng)                

        if tunning:
                test_mask = division_mask == fold
                train_mask = division_mask < n_folds
                
                train_mask = train_mask * ~test_mask
        else:
                test_mask = division_mask == (fold+n_folds)
                train_mask = division_mask >= n_folds
                
                train_mask = train_mask * ~test_mask
        
        train_indexes = torch.where(train_mask)[0]
        test_indexes = torch.where(test_mask)[0]
        
        # SUPER WAŻNE!
        uuc_idx = torch.argwhere(data.targets == len(kkc_indexes))
        train_indexes = np.array([i for i in train_indexes if i not in uuc_idx])


        data_test = copy.deepcopy(data)
        data_train = copy.deepcopy(data)

        data_test.reindex(test_indexes)
        data_train.reindex(train_indexes)  
        
        return data_train, data_test

