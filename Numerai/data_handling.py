import pandas as pd
from numerapi import NumerAPI
import os
import json


class DataHandler:
    def __init__(self, version='v4', feature_set='medium', secondary_targets=None, 
                 every_fourth=True, get_dataloader=False, batch_size=64):
        self.api = NumerAPI()
        self.current_round = self.api.get_current_round()
        self.version = version         
        self.download_data()
        self.features, self.extra_features = self.get_feature_set(feature_set)
        self.train_df = pd.read_parquet(f'{version}/train.parquet', 
                                        columns=self.feature_set)
        if every_fourth:
            self.train_df = self.pare_down(self.train_df)

        self.validation_df = pd.read_parquet(f'{version}/validation.parquet', 
                                             columns=self.feature_set)
        self.live_df = pd.read_parquet(f'{version}/live_{self.current_round}.parquet', 
                                       columns=self.feature_set)
        self.target = f'target_nomi_{self.handler.version}_20'
        self.secondary_targets = secondary_targets
        if get_dataloader:
            train_dataset = Dataset(self.train_df, self.features, self.target)
            validation_dataset = Dataset(self.validation_df, self.features, self.target)
            live_dataset = Dataset(self.live_df, self.features, self.target)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
            self.validation_loader = DataLoader(validation_dataset, batch_size=batch_size) 
            self.live_loader = DataLoader(live_dataset, batch_size=batch_size) 
            del train_dataset
            del validation_dataset
            del live_dataset
            gc.collect()

    def download_data(self):
        print('...Fetching Data...')
        self.api.download_dataset(f'{self.version}/train.parquet')
        self.api.download_dataset(f'{self.version}/validation.parquet')
        self.api.download_dataset(f'{self.version}/live.parquet', 
                                  f'{self.version}/live_{self.current_round}.parquet')
        self.api.download_dataset(f'{self.version}/validation_example_preds.parquet')
        self.api.download_dataset(f'{self.version}/features.json')

    def get_feature_set(self, feature_set):
        with open(f'{self.version}/features.json', 'r') as f:
            feature_metadata = json.load(f)
        extra_features = ['era', 'data_type', f'target_nomi_{self.version}_20']
        if feature_set == 'all':
            return (list(feature_metadata['features_stats'].keys()), extra_features)
        return (feature_metadata['feature_sets'][feature_set], extra_features) 


    def get_n_most_corr(self, corrs, n):
        all_eras = corrs.index.sort_values()
        h1_eras = all_eras[:len(all_eras) // 2]
        h2_eras = all_eras[len(all_eras) // 2:]

        h1_corr_means = corrs.loc[h1_eras, :].mean()
        h2_corr_means = corrs.loc[h2_eras, :].mean()

        corr_diffs = h2_corr_means - h1_corr_means
        worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
        return worst_n

    def pare_down(self, df):
        every_fourth_era = training_data['era'].unique()[::4]
        df = df[df['era'].isin(every_fourth_era)]
        return df


class NumeraiDataset:
    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = self.df.iloc[idx, self.features].values
        y = self.df.iloc[idx, self.target].values
        X = T.tensor(X, dtype=T.float32).to(self.device)
        y = T.tensor(y, dtype=T.float32).to(self.device)
        return X, y
        















