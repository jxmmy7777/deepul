from deepul.hw2_helper import *
from train_utils import *
from models import *
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def q1(train_data, test_data, part, dset_id):
    """
    train_data: An (n_train, 2) numpy array of floats
    test_data: An (n_test, 2) numpy array of floats

    (You probably won't need to use the two inputs below, but they are there
     if you want to use them)
    part: An identifying string ('a' or 'b') of which part is being run. Most likely
          used to set different hyperparameters for different datasets
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a numpy array of size (1000, 2) of 1000 samples WITH decoder noise, i.e. sample z ~ p(z), x ~ p(x|z)
    - a numpy array of size (1000, 2) of 1000 samples WITHOUT decoder noise, i.e. sample z ~ p(z), x = mu(z)
    """

    """ YOUR CODE HERE """
    
    if dset_id ==1:
        hyperparams = {'lr': 1e-4, 'batch_size': 32, 'num_epochs': 100}
    else:
        hyperparams = {'lr': 1e-4, 'batch_size': 32, 'num_epochs': 100}
    
    encoder = Encoder(2, 100, 2)
    decoder = Decoder(2, 100, 2)
    model = VAE(encoder, decoder)

    
    train_tensor = torch.tensor(train_data)
    test_tensor = torch.tensor(test_data)
    # Create DataLoader without additional transformations
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=1000, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=1000, shuffle=False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = partial(loss_fn_ELBO, mode="gll")

    #optimizer
    #Training optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])

    train_losses, test_losses = train_and_evaluate(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        hyperparams=hyperparams,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpoint_path=f"homeworks/hw2/results/q1{part}_{dset_id}",
        device=device,
        debug_mode=False
    )


    # Sample from the model
    
    samples_w_noise =  model.sample_with_noise(1000, device=device).cpu().detach().numpy()
    samples_wo_noise = model.sample_without_noise(1000, device=device).cpu().detach().numpy()
    

    return train_losses, test_losses, samples_w_noise, samples_wo_noise



if __name__ == "__main__":
    # Load the data
    
    # q1_save_results('a', 1, q1)
    # q1_save_results('a', 2, q1)
    q1_save_results('b', 1, q1)
    q1_save_results('b', 2, q1)