import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, block_size=2, kernel_size=(3, 3), stride=1, padding=1):
        super(Upsample_Conv2d, self).__init__()
        self.depth_to_space = DepthToSpace(block_size=block_size)
        self.conv = nn.Conv2d(in_dim , out_dim, kernel_size, stride, padding)

    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)
        x = self.depth_to_space(x)
        x = self.conv(x)
        return x

class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, block_size=2, kernel_size=(3, 3), stride=1, padding=1):
        super(Downsample_Conv2d, self).__init__()
        self.space_to_depth = SpaceToDepth(block_size=block_size)
        # Adjust in_dim according to space_to_depth output channel expansion
        self.conv = nn.Conv2d(in_dim , out_dim, kernel_size, stride, padding)

    def forward(self, x):
        x = self.space_to_depth(x)
        x_chunks = x.chunk(4, dim=1)
        summed_chunks = torch.stack(x_chunks).sum(dim=0)

        x = summed_chunks/ 4.0
        x = self.conv(x)
        return x

class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, n_filters=256, kernel_size=(3, 3)):
        super(ResnetBlockUp, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)
        
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nn.ReLU()
        self.residual_conv_up = Upsample_Conv2d(n_filters, n_filters, block_size=2, kernel_size=kernel_size, padding=1)
        
        self.shortcut_conv_up = Upsample_Conv2d(in_dim, n_filters, block_size=2, kernel_size=(1, 1), padding=0)
    def forward(self, x):
        shortcut = self.shortcut_conv_up(x)
        
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = self.relu2(x)
        residual = self.residual_conv_up(x)
        
        return residual + shortcut
class ResBlock(nn.Module):
    def __init__(self, in_dim = 256, n_filters=256, kernel_size=(3, 3)):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)

        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size, padding=1)
    def forward(self, x):
        shortcut = x
        x = self.relu1(x)
        x = self.conv1(x)
        
        x = self.relu2(x)
        x = self.conv2(x)
        
        return x + shortcut
    
class ResnetBlockDown(nn.Module):
    def __init__(self, in_dim=256, n_filters=256, kernel_size=(3, 3)):
        super(ResnetBlockDown, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1, bias=False)
        
        self.relu2 = nn.ReLU()
        self.residual_conv_down = Downsample_Conv2d(n_filters, n_filters, block_size=2, kernel_size=kernel_size, padding=1)
        
        self.shortcut_conv_down = Downsample_Conv2d(in_dim, n_filters, block_size=2, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        shortcut = self.shortcut_conv_down(x)
    
        x = self.relu1(x)
        x = self.conv1(x)
        
        x = self.relu2(x)
        residual = self.residual_conv_down(x)
        
        return residual + shortcut
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.net(x)
    
class Generator_SNGAN(nn.Module):
    def __init__(self, z_dim=128, n_filters=128, image_channels=3):
        super(Generator_SNGAN, self).__init__()
        self.z_dim = z_dim
        self.initial_linear = nn.Linear(z_dim, 4*4*256)
        self.block1 = ResnetBlockUp(256, n_filters)
        self.block2 = ResnetBlockUp(n_filters, n_filters)
        self.block3 = ResnetBlockUp(n_filters, n_filters)
        self.bn = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(n_filters, image_channels, kernel_size=(3, 3), padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        z = self.initial_linear(z)
        z = z.view(z.shape[0], 256, 4, 4)  # Reshape z to the shape expected by the first ResBlock
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        z = self.bn(z)
        z = self.relu(z)
        z = self.final_conv(z)
        output = self.tanh(z)
        return output
    def sample(self, n_samples, device):
        z = torch.randn(n_samples, self.z_dim, device=device)
        return self.forward(z)
    
class Discriminator(nn.Module):
    def __init__(self, n_filters=128):
        super(Discriminator, self).__init__()
        self.block1 = ResnetBlockDown(3, n_filters)
        self.block2 = ResnetBlockDown(n_filters, n_filters)
        self.block3 = ResBlock(n_filters, n_filters)
        self.block4 = ResBlock(n_filters, n_filters)
        self.relu = nn.ReLU()
        self.global_sum_pooling = nn.AdaptiveAvgPool2d(1)  # Emulates global sum pooling
        self.final_linear = nn.Linear(n_filters, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.relu(x) #shape (batch_size, n_filters, 8, 8)
        x = self.global_sum_pooling(x) #shape (batch_size, n_filters, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten
        output = self.final_linear(x)
        return output
