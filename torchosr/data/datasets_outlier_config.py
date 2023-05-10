import torch
from ..data.DataWrapper import DataWrapper
from ..data.OutlierDataset import OutlierDataset
import numpy as np
import copy

def configure_division_outlier(base_dataset, outlier_dataset, repeats, n_openness=None, seed=None, min_known_classes=2):
        """
        Method for obtaining configurations for OSR model evaluation using Outlier protocol. KKC come from base_dataset, UUC from outlier_dataset.

        :type base_dataset: VisionDataset
        :param base_dataset: Dataset describing KKC instances
        
        :type outlier_dataset: VisionDataset
        :param outlier_dataset: Dataset describing UUC instances
        
        :type repeats: int
        :param repeats: Number of randol selections of classes for single openness (KKC/UUC class cardinality)
        
        :type n_openness: int
        :param n_openness: Number of KKC/UUC class cardinality to generate
        
        :type seed: int
        :param seed: Random state
        
        :type min_known_classes: int
        :param min_known_classes: Minimum number of known classes
        
        :rtype: List
        :returns: Lit of dataset configurations -- each containing sets of KKC and UUC -- and their Openness
        """
        trng = torch.random.manual_seed(seed)

        # Get number of classes from base and outlier datasets
        n_classes_base = (base_dataset._n_classes())
        n_classes_out = (outlier_dataset._n_classes())

        # Tables for quantities of KKC and UUC and Opennes dependent of quantities
        openness_n_classes = []
        openness = []
        
        for i in range(min_known_classes,n_classes_base):
                for j in range(1,n_classes_out):
                      openness_n_classes.append([i,j])  
                      openness.append(1 - np.sqrt((2*i)/((i*2)+j)))

        openness = torch.tensor(openness)
        openness_n_classes = torch.tensor(openness_n_classes)
        
        if n_openness is not None:
                # Randomly select n_openness configurations
                # Higher selection probability to configurations with larger number of classes
                p = openness_n_classes.sum(1)
                p = p / p.sum()
                rand_indexes = p.multinomial(num_samples=n_openness, replacement=False, generator=trng)
                
                # Select chosen openness and configurations
                op_choice = openness[rand_indexes]
                op_n_classes_choice = openness_n_classes[rand_indexes]
        else:
                # In n_openness is None select all configurations
                op_choice = openness
                op_n_classes_choice = openness_n_classes

        # Now randomly assign classes
        repeats_config =[]
        for kkc_n, uuc_n in op_n_classes_choice:    
                for r in range(repeats):
                        all_kkc = torch.arange(n_classes_base, dtype=float)
                        kkc = all_kkc.multinomial(num_samples=kkc_n, replacement=False, generator=trng)
                        
                        all_uuc = torch.arange(n_classes_out, dtype=float)
                        uuc = all_uuc.multinomial(num_samples=uuc_n, replacement=False, generator=trng)
                        
                        repeats_config.append((kkc, uuc))
        
        return repeats_config, op_choice


def get_train_test_outlier(base_dataset, outlier_dataset, kkc_indexes, uuc_indexes, root, tunning, fold, seed=1410, n_folds=5):
        """
        Method for obtaining Cross-validation folds using Outlier protocol (KKC obtained from base_dataset and UUC from outlier_dataset).

        :type base_dataset: VisionDataset
        :param base_dataset: Dataset describing KKC instances
        
        :type outlier_dataset: VisionDataset
        :param outlier_dataset: Dataset describing UUC instances
        
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

        # Create dataset containing only KKC
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
        
        # Portion of data for tunning
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
        
        # Assign label to UUC objects
        outliers.targets = [torch.zeros(len(kkc_indexes)+1, dtype=torch.float)
                                         .scatter_(0, torch.tensor(len(kkc_indexes)), value=1)
                                         for t in outliers.targets]
        
        data_test = OutlierDataset(root, data_test, outliers, shuffle=False, onehot=False, unknown_label=len(kkc_indexes))
        
        return data_train, data_test
