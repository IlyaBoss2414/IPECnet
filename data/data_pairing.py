import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def prepare_data_pairing(df):
    list_features = df.select_dtypes(exclude=['object']).columns.tolist()
    list_features.remove("Unnamed: 0")
    df_test = pd.concat([df.loc[(df["PA_idx"] == 1) & (df["PC_idx"] == 3)].copy(), df.loc[(df["PA_idx"] == 4) & (df["PC_idx"] == 2)].copy()])
    df_train_all = df.drop(index=df_test.index)  # Sample 20% for the validation set
    df_train = df_train_all.sample(frac=0.8, random_state = 42)  # Sample 20% for the test set
    df_val = df_train_all.drop(index=df_train.index)  # Sample 


    scaler = StandardScaler()

    df_train_scaled = pd.DataFrame(
        scaler.fit_transform(df_train[list_features]), columns=list_features
    )
    df_val_scaled = pd.DataFrame(
        scaler.transform(df_val[list_features]), columns=list_features
    )
    df_test_scaled = pd.DataFrame(
        scaler.transform(df_test[list_features]), columns=list_features
    )


    # Scale the target variable "phi" using StandardScaler
    scaler_phi = StandardScaler()
    df_train_scaled["phi"] = scaler_phi.fit_transform(df_train[["phi"]]).flatten()
    df_val_scaled["phi"] = scaler_phi.transform(df_val[["phi"]]).flatten()
    df_test_scaled["phi"] = scaler_phi.transform(df_test[["phi"]]).flatten()


    # Reset the indices of the DataFrames before adding the columns
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # Reset the indices of the scaled DataFrames
    df_train_scaled.reset_index(drop=True, inplace=True)
    df_val_scaled.reset_index(drop=True, inplace=True)
    df_test_scaled.reset_index(drop=True, inplace=True)

    # Add ["PA_canon", "PC_canon"] columns
    df_train_scaled["PA_canon"] = df_train["PA_canon"]
    df_val_scaled["PA_canon"] = df_val["PA_canon"]
    df_test_scaled["PA_canon"] = df_test["PA_canon"]

    df_train_scaled["PC_canon"] = df_train["PC_canon"]
    df_val_scaled["PC_canon"] = df_val["PC_canon"]
    df_test_scaled["PC_canon"] = df_test["PC_canon"]



    class CustomDatasetBERT_upd_pare(Dataset):
        def __init__(self, df, PA_features, PC_features, Common_features):

            self.df = df
            self.PA_features = PA_features
            self.PC_features = PC_features
            self.Common_features = Common_features

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):  

            row = self.df.iloc[idx]

            inp_PA = row["PA_canon"]
            inp_PC = row["PC_canon"]

            PA_features = np.array(row[self.PA_features], dtype=np.float32)
            PC_features = np.array(row[self.PC_features], dtype=np.float32)
            Common_features = np.array(row[self.Common_features], dtype=np.float32)

            PA_features_concat = np.concatenate(PA_features, axis=None)
            PC_features_concat = np.concatenate(PC_features, axis=None)
            Common_features_concat = np.concatenate(Common_features, axis=None)

            target = torch.tensor(row["phi"], dtype=torch.float32)

            return inp_PA, inp_PC, PA_features_concat,PC_features_concat, Common_features_concat, target
    
    
    # Original list
    list_features.remove("phi")
    elements = list_features

    # Initialize empty lists for PA, PC, and other elements
    pa_list = []
    pc_list = []
    other_list = []
    # PC_
    # Iterate over each element in the list
    for element in elements:
        if element.startswith("PA_"):
            pa_list.append(element)
        elif element.startswith("PC_"):
            pc_list.append(element)
        elif element.startswith("PС_"):
            pc_list.append(element)

        else:
            other_list.append(element)



    train_set = CustomDatasetBERT_upd_pare(df_train_scaled, pa_list, pc_list, other_list)
    val_set = CustomDatasetBERT_upd_pare(df_val_scaled, pa_list, pc_list, other_list)
    test_set = CustomDatasetBERT_upd_pare(df_test_scaled, pa_list, pc_list, other_list)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)
    
    return train_loader, val_loader, test_loader
