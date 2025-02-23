import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import json
import bisect


class DatasetCapacity(Dataset):
    def __init__(
            self, root_path, flag, size, features, data_path, time_enc, freq
    ):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.time_enc = time_enc
        self.freq = freq
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data.shape[-1]

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        df_stamp = df_raw[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date, unit='ms')
        data_stamp = None

        if self.time_enc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.time_enc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # df_raw drop date column
        self.data = df_raw.drop(labels=['date','device','name'], axis=1)
        self.data_stamp = data_stamp
        index_df = pd.read_csv(os.path.join(self.root_path, f'{self.flag}_data_index.csv'))
        self.data_list = index_df.values

    def __getitem__(self, index):
        device, seq_begin = self.data_list[index]
        seq_end = seq_begin + self.seq_len
        res_begin = seq_end - self.label_len
        res_end = res_begin + self.label_len + self.pred_len

        seq = self.data[seq_begin:seq_end].values
        res = self.data[res_begin:res_end].values
        seq_stamp = self.data_stamp[seq_begin:seq_end]
        res_stamp = self.data_stamp[res_begin:res_end]

        return device, seq, res, seq_stamp, res_stamp

    def __len__(self):
        return len(self.data_list)

