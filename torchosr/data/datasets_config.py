from ..data.DataWrapper import DataWrapper
import numpy as np
import copy
import torch

def configure_division(base_dataset, repeats, n_openness=None, seed=None, min_known_classes=2):
        """
        Method for obtaining configurations for OSR model evaluation using Holdout protocol (both KKC and UUC from single dataset).

        :type base_dataset: VisionDataset
        :param base_dataset: Base dataset
        
        :type repeats: int
        :param repeats: Number of randol selections of classes for single openness (KKC/UUC class cardinality)
        
        :type n_openness: int
        :param n_openness: Number of KKC/UUC class cardinality to generate. If None will return all possible configurations.
        
        :type seed: int
        :param seed: Random state
        
        :type min_known_classes: int
        :param min_known_classes: Minimum number of known classes
        
        :rtype: List
        :returns: Lit of dataset configurations -- each containing sets of KKC and UUC -- and their Openness
        """
        # Set random generator with manual seed
        trng = torch.random.manual_seed(seed)
        
        # Get number of classes from base dataset
        n_classes = (base_dataset._n_classes())

        # Tables for all possible numbers of KKC and UUC classes
        kkc, uuc = torch.triu_indices(n_classes-1, n_classes-1)
        kkc += 1
        uuc = n_classes - 1 - uuc
        
        # Remove all the entries that have less known classes than given minimum (default=2)
        kkc, uuc = kkc[kkc>=min_known_classes], uuc[kkc>=min_known_classes]

        # Calculate openness for possible KKC and UUC
        openness_n_classes = torch.stack((kkc,uuc)).T        
        openness = 1 - torch.sqrt((2*kkc)/((2*kkc)+uuc))

        # Randomly select n_openness configurations from possible KKC, UUC and openness if n_openness is not None
        if n_openness is not None:
                # Assign higher selection probability to sets with larger number of classes
                p = openness_n_classes.sum(1)
                p = p / p.sum()
                rand_indexes = p.multinomial(num_samples=n_openness, replacement=False, generator=trng)

                # Get selected openness and KKC/UUC configurations
                op_choice = openness[rand_indexes]
                op_n_classes_choice = openness_n_classes[rand_indexes]
        else:
                # Id n_openness not specified return all possible configurations
                op_choice = openness
                op_n_classes_choice = openness_n_classes

        # For selected konfigurations randomly assign classes
        repeats_config =[]
        for kkc_n, uuc_n in op_n_classes_choice:    
                for r in range(repeats):
                        all_classes = torch.ones(n_classes)
                        kkc = all_classes.multinomial(num_samples=kkc_n, replacement=False, generator=trng)
                        all_classes[kkc] = 0
                        uuc = all_classes.multinomial(num_samples=uuc_n, replacement=False, generator=trng)
                        
                        repeats_config.append((kkc, uuc))
 
        return repeats_config, op_choice

def get_train_test(base_dataset, kkc_indexes, uuc_indexes, root, tunning, fold, seed=1410, n_folds=5):
        """
        Method for obtaining Cross-validation folds using Holdout protocol (both KKC and UUC from single dataset).

        :type base_dataset: VisionDataset
        :param base_dataset: Base dataset
        
        :type kkc_indexes: List
        :param kkc_indexes: List of labels constituting Known Classes        
        
        :type uuc_indexes: List
        :param uuc_indexes: List of labels constituting Unknown Classes
        
        :type root: string
        :param root: Datasets folder
        
        :type tunning: boolean
        :param tunning: Flag. If True will split 10% of data for tunning, otherwise will split 90% of data.
                
        :type fold: int
        :param fold: Fold index
        
        :type n_folds: int
        :param n_folds: Number of folds
        
        :type seed: int
        :param seed: Random state
        
        :rtype: List
        :returns: Train dataset, Test dataset
        """
        
        trng = torch.random.manual_seed(seed)
        
        classes = np.concatenate((kkc_indexes, uuc_indexes))

        # Generate dataset
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
        
        # Define portion of data that will be considered tunning set
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
        
        # REMOVE UUC FROM TRAINING
        uuc_idx = torch.argwhere(data.targets == len(kkc_indexes))
        train_indexes = np.array([i for i in train_indexes if i not in uuc_idx])

        data_test = copy.deepcopy(data)
        data_train = copy.deepcopy(data)

        data_test.reindex(test_indexes)
        data_train.reindex(train_indexes)  
        
        return data_train, data_test

