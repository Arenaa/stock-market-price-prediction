
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_dataloaders(
        data_dir: str,
        lookback: int,
        batch_size: int,
        train_test_split: float
):
    data = pd.read_csv(data_dir)

    shifted_df = prepare_Dataframe_for_lstm(data, lookback)
    shifted_df_as_np = shifted_df.to_numpy()

    #Normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    x = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    split = int(len(x)  * train_test_split)

    X_train = x[0:split, :]
    X_test = x[split:len(x), :]
    y_train = y[0:split]
    y_test = y[split:]

    X_train = torch.from_numpy(X_train).unsqueeze_(2)
    X_test = torch.from_numpy(X_test).unsqueeze_(2)

    y_train = torch.from_numpy(y_train).unsqueeze_(1)
    y_test = torch.from_numpy(y_test).unsqueeze_(1)

    X_train = X_train.float()
    X_test = X_test.float()
    y_train = y_train.float()
    y_test = y_test.float()


    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_dataloader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      pin_memory=True,
  )

    test_dataloader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=True,
      pin_memory=True,
  )

    return train_dataloader, test_dataloader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def prepare_Dataframe_for_lstm(df, lookback):
    df['Date']  =  pd.to_datetime(df['Date'])
    df.set_index('Date',  inplace=True)

    for i  in range(1,  lookback+1):
        df[f'Close(t-{i})']  =  df['Close'].shift(i)

    df.dropna(inplace=True)
    return df
