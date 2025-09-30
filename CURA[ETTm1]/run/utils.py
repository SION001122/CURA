import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=96, pred_len=24):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = data

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        seq_x = self.data[idx:idx+self.seq_len]
        seq_y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

def create_sequences(data, seq_len, pred_len, target_slice=slice(None)):
    x_seq, y_seq = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i : i + seq_len]
        y = data[i + seq_len : i + seq_len + pred_len, target_slice]
        x_seq.append(x)
        y_seq.append(y)
    x_seq = np.array(x_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)
    return x_seq, y_seq

def load_ettdataset(
    path, seq_len, pred_len, batch_size=64, feature_type="M", target="OT"
):
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    if feature_type == "M":
        target_slice = slice(None)
    else:
        target_idx = df.columns.get_loc(target)
        target_slice = slice(target_idx, target_idx + 1)

    data = df.values.astype(np.float32)

    train_end = 12 * 30 * 24 * 4
    val_end   = train_end + 4 * 30 * 24 * 4
    test_end  = val_end + 4 * 30 * 24 * 4

    assert data.shape[0] >= test_end, f"데이터 row 부족! {data.shape[0]} < {test_end}"


    train_data = data[:train_end]
    val_data   = data[train_end - seq_len : val_end]
    test_data  = data[val_end - seq_len : test_end]

    mean = train_data.mean(axis=0)
    std  = train_data.std(axis=0)

    def standardize(x): return (x - mean) / std

    train_scaled = standardize(train_data)
    val_scaled   = standardize(val_data)
    test_scaled  = standardize(test_data)


    x_train, y_train = create_sequences(train_scaled, seq_len, pred_len, target_slice)
    x_val, y_val     = create_sequences(val_scaled, seq_len, pred_len, target_slice)
    x_test, y_test   = create_sequences(test_scaled, seq_len, pred_len, target_slice)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader, test_loader, mean, std
