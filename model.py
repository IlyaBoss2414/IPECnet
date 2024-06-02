{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/IlyaBoss2414/IPEC/blob/main/model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "execution": {
          "iopub.execute_input": "2024-04-21T10:11:26.948449Z",
          "iopub.status.busy": "2024-04-21T10:11:26.947549Z",
          "iopub.status.idle": "2024-04-21T10:11:26.963080Z",
          "shell.execute_reply": "2024-04-21T10:11:26.962124Z",
          "shell.execute_reply.started": "2024-04-21T10:11:26.948409Z"
        },
        "id": "ezD5u_2Wl-H4",
        "tags": []
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "from transformers import AutoModel, AutoTokenizer\n",
        "\n",
        "\n",
        "class IPEC_net_upd(nn.Module):\n",
        "    def __init__(self, featureslen=10):\n",
        "        super(IPEC_net_upd, self).__init__()\n",
        "        self.featureslen = featureslen\n",
        "        self.device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "        self.tokenizer = AutoTokenizer.from_pretrained(\"kuelumbus/polyBERT\")\n",
        "        self.polyBERT = AutoModel.from_pretrained(\"kuelumbus/polyBERT\")\n",
        "\n",
        "        self.feature_net = nn.Sequential(\n",
        "            nn.Linear(self.featureslen, 128),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(0.5),\n",
        "        )\n",
        "        self.layers_stack_chem = nn.Sequential(\n",
        "            nn.Linear(1200, 128),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(0.5),\n",
        "        )\n",
        "\n",
        "        self.layers_stack_all = nn.Sequential(\n",
        "            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)\n",
        "        )\n",
        "\n",
        "    def mean_pooling(self, x):\n",
        "        encoded_input = self.tokenizer(\n",
        "            x, padding=True, truncation=True, return_tensors=\"pt\"\n",
        "        ).to(self.device)\n",
        "\n",
        "        model_output = self.polyBERT(**encoded_input)\n",
        "        token_embeddings = model_output[0]\n",
        "        attention_mask = encoded_input[\"attention_mask\"]\n",
        "        input_mask_expanded = (\n",
        "            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()\n",
        "        )\n",
        "\n",
        "        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(\n",
        "            input_mask_expanded.sum(1), min=1e-9\n",
        "        )\n",
        "\n",
        "    def forward(self, x, y, concatenated_features):\n",
        "        int_PA, int_PC, concatenated_features = x, y, concatenated_features\n",
        "\n",
        "        out_1 = self.mean_pooling(int_PA)\n",
        "        out_2 = self.mean_pooling(int_PC)\n",
        "\n",
        "        combined_output_chem = torch.cat([out_1, out_2], dim=1)\n",
        "        combined_output_chem = combined_output_chem.to(self.device)\n",
        "\n",
        "        if isinstance(int_PA, str):\n",
        "            int_PA = torch.tensor(int_PA).to(self.device)\n",
        "\n",
        "        if isinstance(int_PC, str):\n",
        "            int_PC = torch.tensor(int_PC).to(self.device)\n",
        "\n",
        "        chem_output = self.layers_stack_chem(combined_output_chem)\n",
        "\n",
        "        fich_output = self.feature_net(concatenated_features)\n",
        "\n",
        "        combined_output_all = torch.cat([chem_output, fich_output], dim=1)\n",
        "        combined_output_all = combined_output_all.to(self.device)\n",
        "\n",
        "        final_output = self.layers_stack_all(combined_output_all)\n",
        "\n",
        "        return final_output"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": [],
      "toc_visible": true,
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "DataSphere Kernel",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.12"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}