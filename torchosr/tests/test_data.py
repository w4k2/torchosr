"""Data tests."""

import torch
import torchosr as to
from torchvision import transforms

# TRANSFORMS
t_mnist = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()])

t_omni = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        to.utils.inverse_transform()])

t_svhn = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()])

t_cifar = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()])

root = 'data'
n_openness = 2
repeats = 2
n_splits = 5
seed = 43


def test_MNIST_Omni_config():
    base = to.data.MNIST_base(root=root, download=True, transform=t_mnist)
    out = to.data.Omniglot_base(root=root, download=True, transform=t_omni)

    config = to.data.configure_division_o(base, out, n_openness, repeats, seed)

    for config_i, (kkc, uuc) in enumerate(config[0]):
            train, test = to.data.get_train_test_o(base, out, kkc, uuc, root=root, tunning=True, fold=1, seed=seed)
            train_data = [train.__getitem__(i)[0] for i in range(len(train))]
            test_data = [test.__getitem__(i)[0] for i in range(len(test))]
            
            for trd_i, trd in enumerate(train_data):
                    for ted_i, ted in enumerate(test_data):
                            if torch.equal(trd, ted):
                                    exit()


def test_MNIST_config():
    base = to.data.MNIST_base(root=root, download=True, transform=t_mnist)
    config = to.data.configure_division(base, n_openness, repeats, seed)

    for config_i, (kkc, uuc) in enumerate(config[0]):
            train, test = to.data.get_train_test(base, kkc, uuc, root=root, tunning=True, fold=1, seed=seed)

            train_data = [train.__getitem__(i)[0] for i in range(len(train))]
            test_data = [test.__getitem__(i)[0] for i in range(len(test))]
            
            for trd_i, trd in enumerate(train_data):
                    for ted_i, ted in enumerate(test_data):
                            if torch.equal(trd, ted):
                                    exit()

def test_trainset():
    base = to.data.MNIST_base(root=root, download=True, transform=t_mnist)
    config = to.data.configure_division(base, n_openness, repeats, seed)

    for config_i, (kkc, uuc) in enumerate(config[0]):
        train, test = to.data.get_train_test(base, kkc, uuc, root=root, tunning=True, fold=1, seed=seed)

        train_labels = [train.__getitem__(i)[1] for i in range(len(train))]
        
        for trd_i, trd in enumerate(train_labels):
                #Czy na pewno w zbiorze treningowym nie znajdzie się UUC
                if (trd[-1]!=0):
                        exit()


def test_SVHN_config():
    base = to.data.SVHN_base(root=root, download=True, transform=t_svhn)
    config = to.data.configure_division(base, n_openness, repeats, seed)

    for config_i, (kkc, uuc) in enumerate(config[0]):
            train, test = to.data.get_train_test(base, kkc, uuc, root=root, tunning=True, fold=3, seed=seed)

            train_data = [train.__getitem__(i)[0] for i in range(len(train))]
            test_data = [test.__getitem__(i)[0] for i in range(len(test))]
            
            for trd_i, trd in enumerate(train_data):
                    for ted_i, ted in enumerate(test_data):
                            if torch.equal(trd, ted):
                                    exit()

def test_CIFAR10_config():
    base = to.data.CIFAR10_base(root=root, download=True, transform=t_cifar)
    config = to.data.configure_division(base, n_openness, repeats, seed)

    for config_i, (kkc, uuc) in enumerate(config[0]):
            train, test = to.data.get_train_test(base, kkc, uuc, root=root, tunning=True, fold=3, seed=seed)

            train_data = [train.__getitem__(i)[0] for i in range(len(train))]
            test_data = [test.__getitem__(i)[0] for i in range(len(test))]
            
            for trd_i, trd in enumerate(train_data):
                    for ted_i, ted in enumerate(test_data):
                            if torch.equal(trd, ted):
                                    exit()

def test_CIFAR100_config():
    base = to.data.CIFAR100_base(root=root, download=True, transform=t_cifar)
    config = to.data.configure_division(base, n_openness, repeats, seed)

    for config_i, (kkc, uuc) in enumerate(config[0]):
            train, test = to.data.get_train_test(base, kkc, uuc, root=root, tunning=True, fold=3, seed=seed)

            train_data = [train.__getitem__(i)[0] for i in range(len(train))]
            test_data = [test.__getitem__(i)[0] for i in range(len(test))]
            
            for trd_i, trd in enumerate(train_data):
                    for ted_i, ted in enumerate(test_data):
                            if torch.equal(trd, ted):
                                    exit()

def test_config_replicability():
    r_seed = 922
    r_openness = 10
    r_repeats = 10
   
    base = to.data.MNIST_base(root=root, download=True, transform=t_mnist)
    
    config1 = to.data.configure_division(base, r_openness, r_repeats, r_seed)
    config2 = to.data.configure_division(base, r_openness, r_repeats, r_seed)
    
    for i in range(len(config1)):
            #kkc
            assert(torch.equal(config1[0][i][0], config2[0][i][0]))
            #uuc
            assert(torch.equal(config1[0][i][1], config2[0][i][1]))
            #openness
            assert(torch.equal(config1[1][i], config2[1][i]))
            

def test_train_test_replicability():
    r_seed = 922
    r_kkc = [0,1,2]
    r_uuc = [3,4,5,6,7]
   
    base = to.data.MNIST_base(root=root, download=True, transform=t_mnist)
    
    train1, test1 = to.data.get_train_test(base, r_kkc, r_uuc, root=root, tunning=True, fold=1, seed=r_seed)
    train2, test2 = to.data.get_train_test(base, r_kkc, r_uuc, root=root, tunning=True, fold=1, seed=r_seed)

    for i in range(len(train1)):
            d1, t1 = train1.__getitem__(i)
            d2, t2 = train2.__getitem__(i)
            
            assert(torch.equal(d1, d2))
            assert(torch.equal(t1, t2))
            
    for i in range(len(test1)):
            d1, t1 = test1.__getitem__(i)
            d2, t2 = test2.__getitem__(i)
            
            assert(torch.equal(d1, d2))
            assert(torch.equal(t1, t2))

def test_tune_validation():
        base = to.data.MNIST_base(root=root, download=True, transform=t_mnist)
        kkc = [1,2]
        uuc = [0,3]

        for fold in range(n_splits):

                train_tunning, test_tunning = to.data.get_train_test(base, kkc, uuc, root=root, tunning=True, fold=fold, seed=seed)
                train_validation, test_validation = to.data.get_train_test(base, kkc, uuc, root=root, tunning=False, fold=fold, seed=seed)

                train_tunning_data = [train_tunning.__getitem__(i)[0] for i in range(len(train_tunning))]
                train_validation_data = [train_validation.__getitem__(i)[0] for i in range(len(train_validation))]
                                
                test_tunning_data = [test_tunning.__getitem__(i)[0] for i in range(len(test_tunning))]
                test_validation_data = [test_validation.__getitem__(i)[0] for i in range(len(test_validation))]

                validation_data = train_validation_data + test_validation_data
                tunning_data = train_tunning_data + test_tunning_data
                
                for vd_i, vd in enumerate(validation_data):
                        for td_i, td in enumerate(tunning_data):
                                if torch.equal(vd, td):
                                        exit()