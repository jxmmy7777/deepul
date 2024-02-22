from deepul.hw2_helper import *
from train_utils import *
from models import *
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from vqvae import VQVAE
from pixelcnn import PixelCNN
def q3(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of VQ-VAE train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of VQ-VAE test losses evaluated once at initialization and after each epoch
    - a (# of training iterations,) numpy array of PiexelCNN prior train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of PiexelCNN prior test losses evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples with values in {0, ... 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in [0, 255]
    """

    """ YOUR CODE HERE """
    
    hyperparams = {'lr': 1e-4, 'num_epochs': 20}
    
    
    model = VQVAE(input_channels=3, K=128, D=256)
  
    train_tensor = (torch.tensor(train_data)*1.0 / 255.0 -0.5) *2
    test_tensor = (torch.tensor(test_data)*1.0 / 255.0 - 0.5) *2
    
    train_tensor = train_tensor.permute(0, 3, 1, 2)
    test_tensor = test_tensor.permute(0, 3, 1, 2)
    # Create DataLoader without additional transformations
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=128, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # load from checkpoint
    # model.load_state_dict(torch.load(f"homeworks/hw2/results/q2b_{dset_id}/model.pth"))

    #optimizer
    #Training optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])

    train_losses, test_losses = train_and_evaluate(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        hyperparams=hyperparams,
        optimizer=optimizer,
        checkpoint_path=f"homeworks/hw2/results/q3_{dset_id}",
        device=device,
        debug_mode=False
    )
    
    vqvae_train_losses = train_losses[:, 0] #extract total loss
    vqvae_test_losses = test_losses[:, 0] #extract total loss
    
    
    #--------------------------------------Train PixelCNN prior ------------------------------------
    # quantized the images using the vqvae
    # B_train = train_data.shape[0]
    # B_test = test_data.shape[0]
    
    # train_quantized = torch.tensor(model.quantize(train_data))[:,None].long()# [B_train,1, 8, 8]
    # test_quantized  = torch.tensor(model.quantize(test_data))[:,None].long()#[B_test,1, 8, 8]
    
    train_quantized = torch.tensor(quantize_in_batches(train_data, model))[:,None].long()
    test_quantized =  torch.tensor(quantize_in_batches(test_data, model))[:,None].long()
    # sos_token = vqvae.n_embeddings
    # append sos token to the start of each sequence
    # train_quantized = torch.cat([sos_token * torch.ones(train_quantized.size(0), 1).long(), train_quantized], dim=1)
    # test_quantized = torch.cat([sos_token * torch.ones(test_quantized.size(0), 1).long(), test_quantized], dim=1)
      
    train_loader_cnn = DataLoader(TensorDataset(train_quantized), batch_size=256, shuffle=True)
    test_loader_cnn = DataLoader(TensorDataset(test_quantized), batch_size=256)
    
    pixel_cnn = PixelCNN((1, 8, 8), model.n_embeddings, n_layers=5).to(device) #Q what is color size?
    optimizer = optim.Adam(pixel_cnn.parameters(), lr=1e-3)
    
    
    
    transformer_train_losses, transformer_test_losses = train_and_evaluate(
      train_loader=train_loader_cnn,
      test_loader=test_loader_cnn,
      model=pixel_cnn,
      hyperparams=hyperparams,
      optimizer=optimizer,
      checkpoint_path=f"homeworks/hw2/results/q3_pixelcnn_{dset_id}",
      device=device,
      debug_mode=False
    )


    # Sample from the pixelcnn
    transformer_train_losses = transformer_train_losses[:, 0] #extract total loss
    transformer_test_losses = transformer_test_losses[:, 0] #extract total loss
    samples = pixel_cnn.sample(100).squeeze() #B, h,w,c
    
    #pass through vae
    decoded_samples = model.decode_np(samples)
    
    
    
    # Reconstruct samples
    test_samples = test_tensor[:50].to(device)
    #should we sample or not?
    paired_treconstruct = np.zeros((100, 3, 32, 32))
    reconstruct_samples = model.reconstruct(test_samples).cpu().detach().numpy()
    paired_treconstruct[::2] = test_samples.cpu().detach().numpy()
    paired_treconstruct[1::2] = reconstruct_samples
    


    return (
      vqvae_train_losses, 
      vqvae_test_losses, 
      transformer_train_losses,
      transformer_test_losses,
      (decoded_samples), #already reconstructed
      process_images(paired_treconstruct)
    )
   
    
def process_images(images):
    images = images.transpose(0, 2, 3, 1)
    
    images = (images/2+0.5) * 255.0
    images = np.clip(images, 0, 255)
    
    # Convert to uint8
    images = images.astype(np.uint8)
    return images
def normalize_images(images):
    return (images / 255.0 - 0.5) * 2.0
  
def quantize_in_batches(data, model, batch_size=5000):
    """Quantize data in batches using the provided VQ-VAE model."""
    quantized_batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        quantized_batch = model.quantize(batch)
        quantized_batches.append(quantized_batch)
    # Use np.concatenate to merge the list of numpy arrays
    return np.concatenate(quantized_batches, axis=0)



if __name__ == "__main__":
    # Load the data
    # q2_save_results('a', 1, q2_a)
    # q2_save_results('a', 2, q2_a)
    # q3_save_results(1, q3)
    q3_save_results(2, q3)