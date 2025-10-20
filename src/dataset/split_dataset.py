# split_dataset.py
import pandas as pd
import numpy as np

class DatasetSplitter:
    def __init__(self, csv_file, splits=(0.8, 0.1, 0.1), seed=42):
        self.data = pd.read_csv(csv_file)
        self.splits = splits
        self.seed = seed

    def split_by_category(self):
        np.random.seed(self.seed)
        train_list, val_list, test_list = [], [], []

        for cat, df_cat in self.data.groupby('category'):
            idx = np.random.permutation(len(df_cat))
            n_train = int(len(df_cat) * self.splits[0])
            n_val   = int(len(df_cat) * self.splits[1])

            train_list.append(df_cat.iloc[idx[:n_train]])
            val_list.append(df_cat.iloc[idx[n_train:n_train+n_val]])
            test_list.append(df_cat.iloc[idx[n_train+n_val:]])

        train_df = pd.concat(train_list).reset_index(drop=True)
        val_df   = pd.concat(val_list).reset_index(drop=True)
        test_df  = pd.concat(test_list).reset_index(drop=True)

        return train_df, val_df, test_df
