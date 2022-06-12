import pandas as pd
from numerapi import NumerAPI
import os
import gc
import json


class DataHandler:
    def __init__(self, version='v4', feature_set='medium', secondary_targets=False, 
                 every_fourth=True, dataloader=None, batch_size=64):
        self.api = NumerAPI()
        self.current_round = self.api.get_current_round()
        self.version = version         
        try:
            self.download_data()
        except:
            print('Maximum number of tries reached. Using existing data if available.')
        self.secondary_targets = secondary_targets
        self.features, self.extra_features= self.get_feature_set(
                feature_set
                )
        self.train_df = pd.read_parquet('data_files/train.parquet', 
                                        columns=self.features + self.extra_features)
        if every_fourth:
            self.train_df = self.pare_down(self.train_df)

        self.validation_df = pd.read_parquet('data_files/validation.parquet', 
                                             columns=self.features + self.extra_features)
        self.live_df = pd.read_parquet(f'data_files/live_{self.current_round}.parquet', 
                                       columns=self.features + self.extra_features)
        self.secondary_targets = self.get_secondary_targets()
        self.target = f'target_nomi_{self.version}_20'


    def download_data(self):
        print('...Fetching Data...')
        self.api.download_dataset(f'{self.version}/train.parquet', 'data_files/train.parquet')
        self.api.download_dataset(f'{self.version}/validation.parquet',
                                   'data_files/validation.parquet')
        self.api.download_dataset(f'{self.version}/live.parquet', 
                                  f'data_files/live_{self.current_round}.parquet')
        self.api.download_dataset(f'{self.version}/validation_example_preds.parquet', 
                                   'data_files/validation_example_preds.parquet')
        self.api.download_dataset(f'{self.version}/features.json', 'data_files/features.json')

    def get_feature_set(self, feature_set):
        with open('data_files/features.json', 'r') as f:
            feature_metadata = json.load(f)
        extra_features = ['era', 'data_type', f'target_nomi_{self.version}_20']
        if feature_set == 'all':
            return (list(feature_metadata['features_stats'].keys()), extra_features)
        return feature_metadata['feature_sets'][feature_set], extra_features

    def get_secondary_targets(self):
        extra_targets = [col for col in self.train_df if col.startswith("target_")]
        extra_targets.remove(f'target_nomi_{self.version}_20')
        return extra_targets 

    def get_corrs(self, df, target):
        feature_corrs = df.groupby('era').apply(
                lambda era: era[self.features]\
                            .corrwith(era[target])
                )
        gc.collect()
        return feature_corrs

    def get_n_most_corr(self, corrs=None, df=None, n=None, target=None):
        if corrs is None:
            corrs = self.get_corrs(df, target)
        all_eras = corrs.index.sort_values()
        h1_eras = all_eras[:len(all_eras) // 2]
        h2_eras = all_eras[len(all_eras) // 2:]

        h1_corr_means = corrs.loc[h1_eras, :].mean()
        h2_corr_means = corrs.loc[h2_eras, :].mean()

        corr_diffs = h2_corr_means - h1_corr_means
        worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
        return worst_n

    def pare_down(self, df):
        every_fourth_era = df['era'].unique()[::4]
        df = df[df['era'].isin(every_fourth_era)]
        return df
