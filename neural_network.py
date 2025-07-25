import torch
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoded = None

        self.encoder = torch.nn.Sequential(

            nn.Conv1d(1, 1, kernel_size=2, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(1, 1, kernel_size=2, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(14 * 1 * 1, 10),
            nn.ReLU(True)

        )

        self.decoder = torch.nn.Sequential(

            nn.Linear(10, 14 * 1 * 1),
            nn.ReLU(True),
            nn.Unflatten(1, (1, 14)),
            nn.ConvTranspose1d(1, 1, kernel_size=2, stride=2, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose1d(1, 1, kernel_size=2, stride=2, padding=1, output_padding=0),
            # Correcting stride/padding
            nn.ReLU(True),

            nn.Sigmoid()  # Sigmoid to ensure the output is between 0 and 1

        )


    def forward(self, x):

        encoder_outputs = []

        # Encoder forward pass
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)
            self.encoded = x

        # Reverse the encoder outputs for skip connections
        encoder_outputs = encoder_outputs[::-1]

        # Decoder forward pass with skip connections
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.ConvTranspose2d) and i < len(encoder_outputs):
                x = x + 0.5 * encoder_outputs[i]
            x = layer(x)

        return x


################################################################################################

class LastLayer(nn.Module):
    def __init__(self, autoencoder):
        super(LastLayer, self).__init__()

        self.autoencoder_output = None
        self.autoencoder = autoencoder.encoder

        self.supervised_part = nn.Sequential(nn.Linear(10, 2))

    def forward(self, x):
        x = self.autoencoder(x)

        self.autoencoder_output = x

        x = x.view(x.size(0), -1)

        x = self.supervised_part(x)

        return x