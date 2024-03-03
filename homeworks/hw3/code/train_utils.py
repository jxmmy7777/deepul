import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

adversarial_loss = torch.nn.BCELoss()

def evaluate_generator_discriminator(generator, discriminator, device):
    #return samples drawn from the generator and the discriminator's predictions on them
     #samples
    z = torch.randn(5000, 1, device=device, dtype=torch.float32)
    samples = generator(z).detach().cpu().numpy().flatten()
    
    #linear spaced
    z_interpolate = torch.linspace(-1, 1, 1000).reshape(-1, 1).to(device)
    linear_samples = generator(z_interpolate)
    
    #discriminator output
    discriminator_output = discriminator(linear_samples).detach().cpu().numpy().flatten()
    linear_samples = linear_samples.detach().cpu().numpy().flatten()
    
    return samples, linear_samples, discriminator_output

def train_gan(generator, discriminator, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn, dataloader, device, epochs=100, debug_mode=False, wgan_gp = False, n_critic = 1, g_scheduler = None, d_scheduler = None):
    """
    Train a GAN consisting of a generator and discriminator with an option for a quick debug iteration.

    Args:
    - generator: The generator model.
    - discriminator: The discriminator model.
    - g_optimizer: Optimizer for the generator.
    - d_optimizer: Optimizer for the discriminator.
    - g_loss_fn: Loss function for the generator.
    - d_loss_fn: Loss function for the discriminator.
    - dataloader: DataLoader for the real data.
    - device: The device to train on ('cuda' or 'cpu').
    - epochs: Number of epochs to train for.
    - debug_mode: If True, runs a single epoch with a limited number of batches for debugging.

    Returns:
    - A tuple of (generator_losses, discriminator_losses) capturing the loss history.
    """

    generator_losses = []
    discriminator_losses = []

    debug_epochs = 1 if debug_mode else epochs
    debug_batches = 10

    for epoch in tqdm(range(debug_epochs), desc="Epochs"):
        g_epoch_loss = 0
        d_epoch_loss = 0
        batch_count = 0

        for real_data  in dataloader:
            if debug_mode and batch_count >= debug_batches:
                break
            real_data = real_data[0].to(device)
    
            batch_size = real_data.shape[0]
            
            # -----------------
            #  Train Generator
            # -----------------
            g_optimizer.zero_grad()
            # z = torch.randn((batch_size,*generator.shape),device=device, dtype = torch.float32)  # generator.input_size needs to match your generator's input size
            fake_data = generator.sample(batch_size, device = device)
            output = discriminator(fake_data)
            g_loss = g_loss_fn(output, torch.ones_like(output, device=device))
            g_loss.backward()
            g_optimizer.step()
            if g_scheduler is not None:
                g_scheduler.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for i in range(n_critic):
                d_optimizer.zero_grad()
                real_output = discriminator(real_data)
                
                fake_output = discriminator(fake_data.detach())
                
                
                if wgan_gp:
                    #add gradient penalty
                    d_loss = d_loss_fn(real_output, fake_output)
                    gp_loss = gradient_penalty(real_data, fake_data, discriminator)
                    d_loss += gp_loss
                else:
                    real_loss = d_loss_fn(real_output, torch.ones_like(real_output, device=device))
                    fake_loss = d_loss_fn(fake_output, torch.zeros_like(fake_output, device=device))
                    d_loss = (real_loss + fake_loss)
                d_loss.backward()
                d_optimizer.step()
                if d_scheduler is not None:
                    d_scheduler.step()

       
           
            
            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()

            batch_count += 1

        generator_losses.append(g_epoch_loss / batch_count)
        discriminator_losses.append(d_epoch_loss / batch_count)

        if epoch == 0:
            samples, samples_interpolate, discriminator_output = evaluate_generator_discriminator(generator, discriminator, device)
        if debug_mode:
            print(f'Debug Mode: Epoch [{epoch+1}/{debug_epochs}], Generator Loss: {g_epoch_loss / batch_count}, Discriminator Loss: {d_epoch_loss / batch_count}')

    return generator_losses, discriminator_losses, samples, samples_interpolate, discriminator_output
def gradient_penalty(real_data, fake_data, discriminator):
    epsilon = torch.rand(real_data.shape[0], 1, 1, 1).to(real_data.device)
    interpolate_data = (epsilon * real_data + (1 - epsilon) * fake_data).clone().detach()
    interpolate_data.requires_grad = True
    d_interpolate = discriminator(interpolate_data)
    gradients = torch.autograd.grad(outputs=d_interpolate,
                                    inputs=interpolate_data,
                                    grad_outputs=torch.ones_like(d_interpolate),
                                    create_graph=True, 
                                    retain_graph=True)[0]
    #batch, channel, height, width
    gradients = gradients.view(gradients.shape[0], -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty