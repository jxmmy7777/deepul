import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_generator_discriminator(generator, discriminator,device, sample_size = 1000):
    #return samples drawn from the generator and the discriminator's predictions on them
     #samples
    samples = generator.sample(sample_size, device).detach().cpu().numpy().flatten()
    
    #linear spaced
    z_interpolate = torch.linspace(-1, 1, 1000).reshape(-1, 1).to(device)
    linear_samples = generator(z_interpolate)
    
    #discriminator output
    discriminator_output = discriminator(linear_samples).detach().cpu().numpy().flatten()
    linear_samples = linear_samples.detach().cpu().numpy().flatten()
    
    return samples, linear_samples, discriminator_output


def gradient_penalty(real_data, fake_data, discriminator):
    epsilon = torch.rand(real_data.shape[0], 1, 1, 1).to(real_data.device)
    #change this to uniform from 0 to 1
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
    gradients_norm =  gradients.norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty, gradients_norm
