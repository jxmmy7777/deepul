from deepul.hw2_helper import *
from train_utils import *
from models import *
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def q2_a(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
      pairs of test images. The output should be those 100 images flattened into
      the specified shape with values in {0, ..., 255}
    """

    """ YOUR CODE HERE """
    
  
    hyperparams = {'lr': 1e-3, 'num_epochs': 10}
    
    encoder = ConvEncoder(latent_dim=16)
    decoder = ConvDecoder(latent_dim=16)
    model = VAE(encoder, decoder)

    
    train_tensor = (torch.tensor(train_data)*1.0 / 255.0 -0.5) *2
    test_tensor = (torch.tensor(test_data)*1.0 / 255.0 - 0.5) *2
    
    train_tensor = train_tensor.permute(0, 3, 1, 2)
    test_tensor = test_tensor.permute(0, 3, 1, 2)
    # Create DataLoader without additional transformations
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=128, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
   
    #optimizer
    #Training optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])

    train_losses, test_losses = train_and_evaluate(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        hyperparams=hyperparams,
        optimizer=optimizer,
        checkpoint_path=f"homeworks/hw2/results/q2_{dset_id}",
        device=device,
        debug_mode=False
    )


    # Sample from the model
    
    samples_w_noise =  model.sample_without_noise(size=100, device=device).cpu().detach().numpy()
    # Reconstruct samples
    test_samples = test_tensor[:50].to(device)
    #should we sample or not?
    reconstruct_samples = model.sample_reconstruct(test_samples,device=device).cpu().detach().numpy()
    paired_treconstruct = np.concatenate([test_samples.cpu().numpy(), reconstruct_samples], axis=0)
    #reshape to pair pair?
    
    #Interpolate samples
    interpolate_samples = model.sample_interpolate(test_samples[:10],test_samples[10:20],interpolate_pt=10, device=device).cpu().detach().numpy()
    

    return train_losses, test_losses, process_images(samples_w_noise), process_images(paired_treconstruct), process_images(interpolate_samples)
def process_images(images):
    images = images.transpose(0, 2, 3, 1)
    
    images = (images/2+0.5) * 255.0
    images = np.clip(images, 0, 255)
    
    # Convert to uint8
    images = images.astype(np.uint8)
    return images




if __name__ == "__main__":
    # Load the data
    # q2_save_results('a', 1, q2_a)
    q2_save_results('a', 2, q2_a)