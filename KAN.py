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

# --- KAN Model ---
class KANStegoNet(nn.Module):
    def __init__(self, layer_sizes=[4, 8, 16, 32], use_srm=True, input_channels=3,
                 num_classes=1, spline_order=3, groups=1):
        super(KANStegoNet, self).__init__()
        self.use_srm = use_srm
        self.srm = SRMLayer() if use_srm else nn.Identity()

        self.layers = nn.Sequential(
            KANConv2DLayer(input_dim=input_channels, output_dim=layer_sizes[0],
                           spline_order=spline_order, kernel_size=3, groups=1,
                           padding=1, stride=1, dilation=1),
            KANConv2DLayer(input_dim=layer_sizes[0], output_dim=layer_sizes[1],
                           spline_order=spline_order, kernel_size=3, groups=groups,
                           padding=1, stride=2, dilation=1),
            KANConv2DLayer(input_dim=layer_sizes[1], output_dim=layer_sizes[2],
                           spline_order=spline_order, kernel_size=3, groups=groups,
                           padding=1, stride=2, dilation=1),
            KANConv2DLayer(input_dim=layer_sizes[2], output_dim=layer_sizes[3],
                           spline_order=spline_order, kernel_size=3, groups=groups,
                           padding=1, stride=1, dilation=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output = nn.Linear(layer_sizes[3], num_classes)
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.srm(x)
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.output(x)
        return x.squeeze(1)
