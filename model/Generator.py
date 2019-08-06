'''
    writer: dororongju
    github: https://github.com/djkim1991/GAN/issues/1
'''
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=100, out_features=512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=28*28),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x:   input tensor    [batch_size * noise_size]
        :return:    output tensor   [batch_size * 1 * 28 * 28]
        """

        x = self.layer1(x)
        x = self.layer2(x)

        return x.view(-1, 1, 28, 28)
