import torch.nn as nn
import torch.nn.functional as F
from dcase2020.model_architectures.anomaly_score_model import AnomalyScoreModel

class BaselineModel(AnomalyScoreModel):

    def __init__(self, input_dim, dim=128):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, dim)
        self.enc1_bn = nn.BatchNorm1d(dim)
        self.enc2 = nn.Linear(dim, dim)
        self.enc2_bn = nn.BatchNorm1d(dim)
        self.enc3 = nn.Linear(dim, dim)
        self.enc3_bn = nn.BatchNorm1d(dim)
        self.enc4 = nn.Linear(dim, dim)
        self.enc4_bn = nn.BatchNorm1d(dim)
        self.enc5 = nn.Linear(dim, 8)
        self.enc5_bn = nn.BatchNorm1d(8)

        self.dec1 = nn.Linear(8, dim)
        self.dec1_bn = nn.BatchNorm1d(dim)
        self.dec2 = nn.Linear(dim, dim)
        self.dec2_bn = nn.BatchNorm1d(dim)
        self.dec3 = nn.Linear(dim, dim)
        self.dec3_bn = nn.BatchNorm1d(dim)
        self.dec4 = nn.Linear(dim, dim)
        self.dec4_bn = nn.BatchNorm1d(dim)
        self.dec5 = nn.Linear(dim, dim)
        self.dec5_bn = nn.BatchNorm1d(dim)

        self.out = nn.Linear(dim, input_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.enc1_bn(self.enc1(x)))
        x = F.relu(self.enc2_bn(self.enc2(x)))
        x = F.relu(self.enc3_bn(self.enc3(x)))
        x = F.relu(self.enc4_bn(self.enc4(x)))
        x = F.relu(self.enc5_bn(self.enc5(x)))

        x = F.relu(self.dec1_bn(self.dec1(x)))
        x = F.relu(self.dec2_bn(self.dec2(x)))
        x = F.relu(self.dec3_bn(self.dec3(x)))
        x = F.relu(self.dec4_bn(self.dec4(x)))
        x = F.relu(self.dec5_bn(self.dec5(x)))

        x = self.out(x)

        return x

    def anomaly_score(self, x, y, loss, device="cuda:0"):
        recon = self.forward(x)
        x = x.view(x.size(0), -1)
        recon = recon.squeeze()
        anomaly_score = loss(recon, x).mean(axis=1)
        return anomaly_score


if __name__ == '__main__':
    import torch
    x = torch.randn((8, 10, 64)) # batch_size x input_dim1 x input_dim2
    net = BaselineModel(640)
    print(net)
    print(net(x))