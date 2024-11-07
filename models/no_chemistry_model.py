import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class IPEC_net_priz(nn.Module):
    def __init__(self, featureslen=10):
        super(IPEC_net_priz, self).__init__()

        self.featureslen = featureslen
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_net = nn.Sequential(
            nn.Linear(self.featureslen, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, x, y, concatenated_features):

        int_PA, int_PC, concatenated_features = x, y, concatenated_features
        fich_output = self.feature_net(concatenated_features)

        return fich_output