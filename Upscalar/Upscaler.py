import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class Upscaler(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=64, num_blocks=8):
        super().__init__()
        self.input = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.residual = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.middle = nn.Conv2d(channels, channels, 3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.input(x)
        residual = x
        x = self.residual(x)
        x = self.middle(x)
        x = x + residual
        x = self.upscale(x)
        x = self.output(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Upscaler().to(device)
    x = torch.randn(4, 3, 64, 64, device=device)
    y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))