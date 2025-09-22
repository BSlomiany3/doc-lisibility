from torch import nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels//4, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        return self.block(x)
    
class VGGLikeNN(nn.Module):
    def __init__(self, channels_param, num_classes):
        super().__init__()
        
        self.features = nn.ModuleList()
        for in_chan, out_chan in channels_param:
            self.features.append(VGGBlock(in_chan, out_chan))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x, log_activations=False):
        activations = {}

        for i, block in enumerate(self.features):
            x = block(x)
            if log_activations:
                activations[f'block{i+1}_out'] = x

        logits = self.classifier(x)
        if log_activations:
            activations['logits'] = logits
            return logits, activations
        
        return logits