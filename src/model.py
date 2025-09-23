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


class ConvBlock(nn.Module):
    """(Conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class UNetPP(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, deep_supervision=False, base_filters=64):
        super(UNetPP, self).__init__()
        self.deep_supervision = deep_supervision

        nb_filter = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8, base_filters * 16]

        # Encoder
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        self.pool = nn.MaxPool2d(2)

        # Decoder (nested)
        self.up1_0 = Up(nb_filter[1], nb_filter[0])
        self.up2_0 = Up(nb_filter[2], nb_filter[1])
        self.up3_0 = Up(nb_filter[3], nb_filter[2])
        self.up4_0 = Up(nb_filter[4], nb_filter[3])

        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[0], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[1], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[2], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[3], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[0], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[1], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[2], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[0], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[1], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[0], nb_filter[0])

        # Deep supervision heads
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Decoder path (nested connections)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_0(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_0(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_0(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_0(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_0(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_0(x1_3)], 1))

        # Output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output


class SpectralPixelNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super(SpectralPixelNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=16),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=32, out_features=16),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNetPP(in_channels=5, out_channels=1)
    model = model.to(device)
    x = torch.randn(1, 5, 128, 128)
    x = x.to(device)
    out = model(x)
    print(out.shape)

    summary(model, input_size=(5, 128, 128))
