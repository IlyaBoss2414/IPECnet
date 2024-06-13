def prepare_data(df):
    list_features = df.select_dtypes(exclude=['object']).columns.tolist()
    list_features = list_features.remove("Unnamed: 0")
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


    class CustomDatasetBERT(Dataset):
        def __init__(self, df, list_features):
            self.df = df

            self.list_features = list_features

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):  # как ообращаться

            row = self.df.iloc[idx]

            inp_PA = row["PA_canon"]
            inp_PC = row["PC_canon"]

            features = np.array(row[self.list_features], dtype=np.float32)

            concatenated_features = np.concatenate(features, axis=None)

            target = torch.tensor(row["phi"], dtype=torch.float32)

            return inp_PA, inp_PC, concatenated_features, target



    train_set = CustomDatasetBERT(df_train_scaled, list_features)
    val_set = CustomDatasetBERT(df_val_scaled, list_features)
    test_set = CustomDatasetBERT(df_test_scaled, list_features)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)
    
    return train_loader, val_loader, test_loader