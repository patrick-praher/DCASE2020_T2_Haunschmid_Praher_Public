import torch.nn as nn
from dcase2020.model_architectures.anomaly_score_model import AnomalyScoreModel

class FullyConvolutionalAE(AnomalyScoreModel):
    # Source: https://stackoverflow.com/questions/58198305/why-is-my-fully-convolutional-autoencoder-not-symmetric?rq=1
    def __init__(self, channel_multiplier = 4, out_length = 8):
        super(FullyConvolutionalAE, self).__init__()

        self.encoder = nn.Sequential(
            # conv 1
            nn.Conv2d(in_channels=1, out_channels=4 * channel_multiplier, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4 * channel_multiplier),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv 2
            nn.Conv2d(in_channels=4 * channel_multiplier, out_channels=8 * channel_multiplier, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8 * channel_multiplier),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv 3
            nn.Conv2d(in_channels=8 * channel_multiplier, out_channels=16 * channel_multiplier, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16 * channel_multiplier),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv 4
            nn.Conv2d(in_channels=16 * channel_multiplier, out_channels=32 * channel_multiplier, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32 * channel_multiplier),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # # # conv 5
            # nn.Conv2d(in_channels=32 * channel_multiplier, out_channels=64 * channel_multiplier, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64 * channel_multiplier),
            # nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # # conv 6
            # nn.ConvTranspose2d(in_channels=64 * channel_multiplier, out_channels=32 * channel_multiplier, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(32 * channel_multiplier),
            # nn.ReLU(),
            #
            # # conv 7
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=32 * channel_multiplier, out_channels=16 * channel_multiplier, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16 * channel_multiplier),
            nn.ReLU(),

            # conv 8
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=16 * channel_multiplier, out_channels=8 * channel_multiplier, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8 * channel_multiplier),
            nn.ReLU(),

            # conv 9
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=8 * channel_multiplier, out_channels=4 * channel_multiplier, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4 * channel_multiplier),
            nn.ReLU(),

            # conv 10 out
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=4 * channel_multiplier, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.Linear(out_length, out_length)  # depends on depth
            # ,
            # nn.Softmax()    # multi-class classification
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def anomaly_score(self, x, y, loss, device="cuda:0"):
        recon = self.forward(x)
        # x = x.unsqueeze(1)
        recon = recon.squeeze()
        anomaly_score = loss(recon, x).mean(axis=1).mean(axis=1)
        return anomaly_score

class Conv2DAE(nn.Module):
    def __init__(self):
        super().__init__()

        # input: batch x 3 x 32 x 32 -> output: batch x 16 x 16 x 16
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1), # batch x 16 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        self.unpool = nn.MaxUnpool2d(2, stride=2, padding=0)

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        # print(x.size())
        x = x.unsqueeze(1)
        # print(x.size())
        out, indices = self.encoder(x)
        # print("encoder", out.size())
        out = self.unpool(out, indices)
        # print("unpool", out.size())
        out = self.decoder(out)
        # print("decoder", out.size())
        return out


class ConvDAE(nn.Module):
    def __init__(self):
        super().__init__()

        # input: batch x 3 x 32 x 32 -> output: batch x 16 x 16 x 16
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), # batch x 16 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        self.unpool = nn.MaxUnpool2d(2, stride=2, padding=0)

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        print(x.size())
        out, indices = self.encoder(x)
        print("encoder", out.size())
        out = self.unpool(out, indices)
        print("unpool", out.size())
        out = self.decoder(out)
        print("decoder", out.size())
        return out


if __name__ == '__main__':
    import torch
    # more autoencoders:
    # - https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    # - https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/convolutional-autoencoder/Convolutional_Autoencoder_Solution.ipynb
    x1 = torch.randn((17, 128, 16)) # batch_size x input_dim1 x input_dim2
    x = torch.randn((17, 3, 32, 32))
    net = FullyConvolutionalAE(channel_multiplier=4, out_length=x1.shape[2])
    print(net)
    out = net(x1)
    print(out.shape)
    # net = Conv2DAE()
    # print(net)
    # out = net(x1)