import torch
import torch.nn as nn
import torchvision.transforms as transforms

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU()
            )

        self.encoder = nn.Sequential(
            conv_block(in_channels, 64),
            nn.MaxPool2d(2),
            conv_block(64, 128),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            conv_block(128, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            conv_block(64, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)

if __name__ == "__main__":
    model = UNet()
    print(model)
