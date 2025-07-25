import torch
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoded = None

        self.encoder = torch.nn.Sequential(

            nn.Conv1d(1, 1, kernel_size=2, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(1, 1, kernel_size=2, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(6 * 1 * 1, 3),
            nn.ReLU(True)

        )



        self.decoder = torch.nn.Sequential(

            nn.Linear(3, 6 * 1 * 1),
            nn.ReLU(True),
            nn.Unflatten(1, (1, 6)),
            nn.ConvTranspose1d(1, 1, kernel_size=2, stride=2, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose1d(1, 1, kernel_size=2, stride=2, padding=1, output_padding=0),
            # Correcting stride/padding
            nn.ReLU(True),

            nn.Sigmoid()  # Sigmoid to ensure the output is between 0 and 1

        )


    def forward(self, x):
        self.encoded = self.encoder(x)
        decoded = self.decoder(self.encoded)
        return decoded


################################################################################################

class LastLayer(nn.Module):
    def __init__(self, autoencoder):
        super(LastLayer, self).__init__()

        self.autoencoder_output = None
        self.autoencoder = autoencoder.encoder

        self.supervised_part = nn.Sequential(nn.Linear(3, 2))

    def forward(self, x):
        x = self.autoencoder(x)

        self.autoencoder_output = x

        x = x.view(x.size(0), -1)

        x = self.supervised_part(x)

        return x