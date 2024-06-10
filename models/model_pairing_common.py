import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class IPEC_net_upd_pare(nn.Module):
    def __init__(self, PA_featureslen = 10, PC_featureslen = 10, Common_featureslen = 10):
        super(IPEC_net_upd_pare, self).__init__()

        self.PA_featureslen = PA_featureslen
        self.PC_featureslen = PC_featureslen
        self.Common_featureslen = Common_featureslen

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("kuelumbus/polyBERT")
        self.polyBERT = AutoModel.from_pretrained("kuelumbus/polyBERT")

        
        self.feature_PA = nn.Sequential(
            nn.Linear(self.PA_featureslen + self.Common_featureslen, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.feature_PC = nn.Sequential(
            nn.Linear(self.PC_featureslen + self.Common_featureslen, 64),
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

    def mean_pooling(self, x):
        encoded_input = self.tokenizer(
            x, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        model_output = self.polyBERT(**encoded_input)
        token_embeddings = model_output[0]
        attention_mask = encoded_input["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, x, y, PA_features, PC_features, Common_features):
        int_PA, int_PC, PA_features, PC_features, Common_features = x, y, PA_features, PC_features, Common_features

        out_1 = self.mean_pooling(int_PA)
        out_2 = self.mean_pooling(int_PC)

        # Convert input strings to tensors if necessary
        if isinstance(int_PA, str):
            int_PA = torch.tensor(int_PA).to(self.device)

        if isinstance(int_PC, str):
            int_PC = torch.tensor(int_PC).to(self.device)

        PA_chem_output = self.PA_chem(out_1)
        PC_chem_output = self.PC_chem(out_2)
        
        PA_combined_feat = torch.cat([PA_features, Common_features], dim=1)
        PA_features_output = self.feature_PA(PA_combined_feat)
        
        PC_combined_feat = torch.cat([PC_features, Common_features], dim=1)        
        PC_features_output = self.feature_PC(PC_combined_feat)
        
        PA_combined = torch.cat([PA_chem_output, PA_features_output], dim=1)
        PA_combined = PA_combined.to(self.device)
        PA_combined_output = self.PA_all(PA_combined)
  
        PC_combined = torch.cat([PC_chem_output, PC_features_output], dim=1)
        PC_combined = PC_combined.to(self.device)
        PC_combined_output = self.PC_all(PC_combined)
        
        combined_output_all = torch.cat([PA_combined_output, PC_combined_output, Common_features], dim=1)
        combined_output_all = combined_output_all.to(self.device)

        final_output = self.layers_stack_all(combined_output_all)

        return final_output