# --coding:utf-8--
import torch.nn as nn

class GAF_MLP(nn.Module):
    def __init__(self):

        super(GAF_MLP, self).__init__()

        feats = [16, 32, 64, 128, 256, 512, 1024]

        self.GAF_block = nn.Sequential(
            nn.Linear(219, feats[4]),
            nn.ReLU(),
            nn.Linear(feats[4], feats[3]),
            nn.ReLU(),
            nn.Linear(feats[3], 1)
        )

    def forward(self, GAF):
        x = self.GAF_block(GAF)
        return x
    