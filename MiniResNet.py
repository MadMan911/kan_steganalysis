class SRMLayer(nn.Module):
    def __init__(self):
        super(SRMLayer, self).__init__()
        kernel1 = [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
        kernel2 = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
        kernel3 = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
        kernels = [kernel1, kernel2, kernel3]
        filters = torch.tensor(kernels, dtype=torch.float32).unsqueeze(1)
        self.weight = nn.Parameter(filters, requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MiniResNetStego(nn.Module):
    def __init__(self):
        super(MiniResNetStego, self).__init__()
        self.srm = SRMLayer()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.layer1 = ResidualBlock(8, 8)
        self.layer2 = ResidualBlock(8, 16, stride=2)
        self.layer3 = ResidualBlock(16, 32, stride=2)
        self.layer4 = ResidualBlock(32, 32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.25)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.srm(x)
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x.squeeze(1)
