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


class MiniStegoNet(nn.Module):
    def __init__(self):
        super(MiniStegoNet, self).__init__()
        self.srm = SRMLayer()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.25)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.srm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x.squeeze(1)
