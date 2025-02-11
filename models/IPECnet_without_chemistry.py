import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class IPEC_net_pairing_priz(nn.Module):
    def __init__(self, PA_featureslen = 10, PC_featureslen = 10, Common_featureslen = 10):
        super(IPEC_net_pairing_priz, self).__init__()

        self.PA_featureslen = PA_featureslen
        self.PC_featureslen = PC_featureslen
        self.Common_featureslen = Common_featureslen

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feature_PA = nn.Sequential(
            nn.Linear(self.PA_featureslen, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.feature_PC = nn.Sequential(
            nn.Linear(self.PC_featureslen, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        

        self.PA_chem = nn.Sequential(
            nn.Linear(600, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.PC_chem = nn.Sequential(
            nn.Linear(600, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        
        self.PA_all = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.PC_all = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.layers_stack_all = nn.Sequential(
            nn.Linear(128 + self.Common_featureslen, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )


    def forward(self, x, y, PA_features, PC_features, Common_features):
        int_PA, int_PC, PA_features, PC_features, Common_features = x, y, PA_features, PC_features, Common_features

        # Convert input strings to tensors if necessary
        if isinstance(int_PA, str):
            int_PA = torch.tensor(int_PA).to(self.device)

        if isinstance(int_PC, str):
            int_PC = torch.tensor(int_PC).to(self.device)

        PA_features_output = self.feature_PA(PA_features)
        PC_features_output = self.feature_PC(PC_features)
        
        
        combined_output_all = torch.cat([PA_features_output, PC_features_output, Common_features], dim=1)
        combined_output_all = combined_output_all.to(self.device)

        final_output = self.layers_stack_all(combined_output_all)
        

        return final_output
