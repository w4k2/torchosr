from torch.utils.data import DataLoader
import torchosr as osr
import torch
from tqdm import tqdm
from torchosr.data.datasets_config import configure_division, get_train_test
from torchosr.models.architectures import fc_lower_stack
from torchvision import transforms
from torchosr.utils.base import inverse_transform

t_mnist = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()])

t_omni = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            inverse_transform()])

# Modelling parameters
learning_rate = 1e-3
batch_size = 64
epochs = 128

# Evaluation parameters
repeats = 5         # openness repeats 
n_splits = 5        # classical validation splits
n_openness = 5      # openness values
root='data'

# Load dataset
base = osr.data.MNIST_base(root=root, download=True, transform=t_mnist)

config, openness = configure_division(base, n_openness, repeats, seed=4334)

n_methods = 2
n_measures = 4
results = torch.full((n_measures, len(config), n_splits, n_methods, epochs), torch.nan)

pbar = tqdm(total=len(config)*n_splits*n_methods*epochs)

# Iterating configurations
for config_idx, (kkc, uuc) in enumerate(config):
        print('# Configuration %i [openness %.3f]' % (config_idx, openness[config_idx // repeats]), 
              'known:', kkc.numpy(), 
              'unknown:', uuc.numpy())
        # Iterate divisions
        for fold in range(n_splits):
            train_data, test_data = get_train_test(base,
                                                   kkc, uuc,
                                                   root='data',
                                                   tunning=False,
                                                   fold=fold,
                                                   seed=1411,
                                                   n_folds=n_splits)
            
            train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
            
            methods = [
            osr.models.Softmax(lower_stack=fc_lower_stack(depth=1, img_size_x=28, n_out_channels=64), n_known=len(kkc), epsilon=0.7),
            osr.models.Openmax(lower_stack=fc_lower_stack(depth=1, img_size_x=28, n_out_channels=64), n_known=len(kkc), epsilon=0.5),
            ]
            
            for model_id, model in enumerate(methods):                         
                # Initialize loss function
                loss_fn = torch.nn.CrossEntropyLoss()

                # Initialize optimizer
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            
                for t in range(epochs):
                    # Train
                    model.train(train_data_loader, loss_fn, optimizer)
                    
                    # Test
                    inner_score, outer_score, hp_score, overall_score = model.test(test_data_loader, loss_fn)
                    results[0, config_idx, fold, model_id, t] = inner_score
                    results[1, config_idx, fold, model_id, t] = outer_score
                    results[2, config_idx, fold, model_id, t] = hp_score
                    results[3, config_idx, fold, model_id, t] = overall_score
             
                    
                    pbar.update(1)
                        
                print(config_idx,fold, model_id, '\n', results[:,config_idx, fold, model_id])     
pbar.close()