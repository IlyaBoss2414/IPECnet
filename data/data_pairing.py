import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def prepare_data_pairing(df):

    list_features = df_train.select_dtypes(exclude=['object']).columns.tolist()
    list_features.remove("phi")
    list_features.remove("label")

    scaler = StandardScaler()

    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    df_train_scaled[list_features] = scaler.fit_transform(df_train[list_features])
    df_val_scaled[list_features] = scaler.transform(df_val[list_features])
    df_test_scaled[list_features] = scaler.transform(df_test[list_features])


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

            target = torch.tensor(row["label"], dtype=torch.float32)

            return inp_PA, inp_PC, PA_features_concat,PC_features_concat, Common_features_concat, target

    
    # Original list
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
            


    # Calculate sample weights based on the frequency of targets
    def calculate_sample_weights(df):
        target_counts = df["label"].value_counts().to_dict()  # Count occurrences of each target
        total_samples = len(df)

        # Assign weights inversely proportional to the frequency
        weights = df["label"].map(lambda x: 1.0 / target_counts[x])

        # Normalize weights to sum to 1
        weights /= weights.sum()

        return weights.values

    # Assign weights to the train dataset
    train_weights = calculate_sample_weights(df_train_scaled)

    # Create a WeightedRandomSampler for the DataLoader
    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

    # Create the DataLoader with the WeightedRandomSampler


    train_set = CustomDatasetBERT_upd_pare(df_train_scaled, pa_list, pc_list, other_list)
    val_set = CustomDatasetBERT_upd_pare(df_val_scaled, pa_list, pc_list, other_list)
    test_set = CustomDatasetBERT_upd_pare(df_test_scaled, pa_list, pc_list, other_list)

    batch_size_all = 32
    train_loader = DataLoader(train_set, batch_size = batch_size_all, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size_all, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size_all, shuffle=False)

    
    return train_loader, val_loader, test_loader