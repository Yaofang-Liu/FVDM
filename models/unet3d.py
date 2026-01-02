import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        
        # Contracting Path
        self.enc_conv0 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.enc_conv1 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2, stride=2)
        
        # Expanding Path
        self.up_conv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.up_conv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.final_conv = nn.Conv3d(32, 16, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        x1 = F.relu(self.enc_conv0(x))
        x1p = self.pool(x1)
        x2 = F.relu(self.enc_conv1(x1p))
        x2p = self.pool(x2)
        x3 = F.relu(self.enc_conv2(x2p))
        
        # Expanding Path
        x_up = self.up_conv2(x3)
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = F.relu(self.dec_conv2(x_up))
        
        x_up = self.up_conv1(x_up)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = F.relu(self.dec_conv1(x_up))
        
        x_up = self.final_conv(x_up)
        
        return x_up

# # Model instantiation and example forward pass
# model = UNet3D()
# input_tensor = torch.randn(8, 16, 4, 32, 32)  # Example input tensor
# output_tensor = model(input_tensor)

# print("Input shape:", input_tensor.shape)
# print("Output shape:", output_tensor.shape)
# ipdb.set_trace()