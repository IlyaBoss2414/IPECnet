import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class IPEC_net_upd(nn.Module):
    def __init__(self, featureslen=10):
        super(IPEC_net_upd, self).__init__()
        
        self.featureslen = featureslen
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("kuelumbus/polyBERT")
        self.polyBERT = AutoModel.from_pretrained("kuelumbus/polyBERT")

        self.feature_net = nn.Sequential(
            nn.Linear(self.featureslen, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.layers_stack_chem = nn.Sequential(
            nn.Linear(1200, 600),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(600, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        

        self.layers_stack_all = nn.Sequential(
            nn.Linear(256, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1),
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

    def forward(self, x, y, concatenated_features):
        int_PA, int_PC, concatenated_features = x, y, concatenated_features

        out_1 = self.mean_pooling(int_PA)
        out_2 = self.mean_pooling(int_PC)

        combined_output_chem = torch.cat([out_1, out_2], dim=1)
        combined_output_chem = combined_output_chem.to(self.device)

        if isinstance(int_PA, str):
            int_PA = torch.tensor(int_PA).to(self.device)

        if isinstance(int_PC, str):
            int_PC = torch.tensor(int_PC).to(self.device)

        chem_output = self.layers_stack_chem(combined_output_chem)

        fich_output = self.feature_net(concatenated_features)

        combined_output_all = torch.cat([chem_output, fich_output], dim=1)
        combined_output_all = combined_output_all.to(self.device)

        final_output = self.layers_stack_all(combined_output_all)

        return final_output
