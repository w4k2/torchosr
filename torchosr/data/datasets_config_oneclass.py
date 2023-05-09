import numpy as np
import torch

def configure_oneclass_division(base_dataset, repeats, n_openness=None, seed=None):
        """
        Method for obtaining configurations for OSR model evaluation using Holdout protocol (both KKC and UUC from single dataset) for One-class classification. Set of KKC always contains a single class.

        :type base_dataset: VisionDataset
        :param base_dataset: Base dataset
        
        :type repeats: int
        :param repeats: Number of randol selections of classes for single openness (KKC/UUC class cardinality)
        
        :type n_openness: int
        :param n_openness: Number of KKC/UUC class cardinality to generate. In None will return all possible configurations.
        
        :type seed: int
        :param seed: Random state
        
        :rtype: List
        :returns: List of dataset configurations -- each containing sets of KKC and UUC -- and their Openness
        """
        
        # Set random generator with manual seed
        trng = torch.random.manual_seed(seed)
        
        # Get number of classes from base dataset
        n_classes = (base_dataset._n_classes())

        # Tables for all possible numbers of KKC and UUC classes
        openness_n_classes = []
        openness = []
        
        for j in range(1,n_classes-1):
                openness_n_classes.append([1,j])
                openness.append(1 - np.sqrt((2)/(2+j)))
                
        openness = torch.tensor(openness)
        openness_n_classes = torch.tensor(openness_n_classes)
        
        # Jeżeli mamy n_openness losujemy n konfiguracji
        if n_openness is not None:
                # Faworyzujemy te podzbiory, w których sumaryczna liczba klas jest duża
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

