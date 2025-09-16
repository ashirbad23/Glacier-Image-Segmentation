import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class U_Net(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, bands=[32, 64, 128, 256]):
        super(U_Net, self).__init__()

        self.enc1 = DoubleConv(in_channels, bands[0])
        self.enc2 = DoubleConv(bands[0], bands[1])
        self.enc3 = DoubleConv(bands[1], bands[2])
        self.enc4 = DoubleConv(bands[2], bands[3])

        self.bottle_neck = DoubleConv(bands[3], bands[3] * 2)

        self.upconv4 = nn.ConvTranspose2d(bands[3] * 2, bands[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(bands[3] * 2, bands[3])

        self.upconv3 = nn.ConvTranspose2d(bands[3], bands[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(bands[2] * 2, bands[2])  # FIXED

        self.upconv2 = nn.ConvTranspose2d(bands[2], bands[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(bands[1] * 2, bands[1])  # FIXED

        self.upconv1 = nn.ConvTranspose2d(bands[1], bands[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(bands[0] * 2, bands[0])  # FIXED

        self.final_conv = nn.Conv2d(bands[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        b = self.bottle_neck(F.max_pool2d(e4, 2))

        d4 = self.upconv4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.sigmoid(self.final_conv(d1))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = U_Net(in_channels=5, out_channels=1)
    model = model.to(device)
    x = torch.randn(1, 5, 512, 512)
    x = x.to(device)
    out = model(x)
    print(out.shape)

    summary(model, input_size=(5, 512, 512))
