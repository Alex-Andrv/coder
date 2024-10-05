import torch
import torch.nn as nn

from src.modeling.base import BaseModel


class CustomRegularizationBaseAutoEncoder(BaseModel):
    def __init__(self, model_name='aandreev_ae'):
        super(CustomRegularizationBaseAutoEncoder, self).__init__(model_name)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 3, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x, b_t):
        encoder_output = self.encoder(x)

        maxt = encoder_output.amax(dim=(1, 2, 3), keepdim=True)
        # Use torch.max to get the maximum value of e3
        e3 = encoder_output / maxt      # Divide e3 by maxt
        e3 = torch.clamp(e3, 0, 0.9999999)  # Use torch.clamp to clip values in e3
        e3 *= 2 ** b_t        # Use the power operator with PyTorch
        
        e4 = e3 / (2 ** b_t)    # Same for e4
        e4 *= maxt              # Multiply e4 by maxt


        x = self.decoder(encoder_output)
        if self.training:
            return x, e3
        else:
            return x

def regularization_loss(e3):
    # Latent loss based on the sinusoidal transformation of e3 layer output
    pi_value = 3.141592653589793  # Define Pi as a constant in PyTorch

    # Use PyTorch operations for tensor calculations
    latent_loss = torch.mean(((torch.sin(e3 * 2.0 * pi_value - (3 * pi_value / 2.0)) + 1) / 2) ** 2)

    return latent_loss
