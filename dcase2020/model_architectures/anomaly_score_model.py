import torch.nn as nn


class AnomalyScoreModel(nn.Module):
    def __init__(self):
        super(AnomalyScoreModel, self).__init__()

    def anomaly_score(self, x, y, loss, device="cuda:0"):
        raise NotImplementedError("Please implement anomaly_score")