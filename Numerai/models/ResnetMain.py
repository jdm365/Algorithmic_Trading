import pandas as pd
import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from ModelArchitectures import ResnetMain
sys.path.append('../')
from data_handling import DataHandler


class NumeraiDatasetMain:
    def __init__(self, n_features=75, dataset='train'):
        self.data_handler = DataHandler(
            feature_set='medium',
            secondary_targets=True
            )
        self.df = self.get_df(dataset)
        self.all_targets = self.data_handler.secondary_targets + [self.data_handler.target]
        self.n_features = n_features
        self.corrs_list = self.get_corrs_list() 
        self.features_list = self.get_features_list()
        self.all_eras = self.df.era.unique()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.df.era.unique())

    def __getitem__(self, idx):
        X = self.get_input(idx)
        df = self.df.loc[self.df.era == self.all_eras[idx]] 
        y = df.loc[:, self.data_handler.target].values
        y = T.tensor(y, dtype=T.float32).to(self.device)
        return X, y

    def get_input(self, idx):
        df = self.df.loc[self.df.era == self.all_eras[idx]] 
        X = T.zeros((len(self.features_list), len(df.reset_index().id.unique()), self.n_features))
        print(X.shape[1])
        for i, features in enumerate(self.features_list):
            X[i, :, :] = T.tensor(df.loc[:, features].values, dtype=T.float32)
        return X

    def get_corrs_list(self):
        corrs = []
        for target in self.all_targets:
            corrs.append(self.data_handler.get_corrs(self.df, target))
        return corrs
        
    def get_features_list(self): 
        features = []
        for corr_df in self.corrs_list:
            features.append(
                self.data_handler.get_n_most_corr(corrs=corr_df, n=self.n_features)
                )
        return features

    def get_df(self, dataset):
        if dataset == 'train':
            df = self.data_handler.train_df
        elif dataset == 'validation':
            df = self.data_handler.validation_df
        elif dataset == 'live':
            df = self.data_handler.live_df
        return df

        

class ResnetMainRunner(nn.Module):
    def __init__(
            self,
            in_features=20,
            n_features=75, 
            block_dims=512, 
            kernel_sizes=[(1, 7), (1, 7), (1, 5), (1, 5), (1, 3)],
            batch_size=32,
            lr=1e-3
        ):
        super(ResnetMainRunner, self).__init__()
        self.model = ResnetMain(
                in_features=in_features,
                block_dims=block_dims, 
                kernel_sizes=kernel_sizes,
                n_features=n_features
                )
        self.batch_size = batch_size
        self.n_features = n_features
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def train(self, n_epochs, save_model=True):
        dataset = NumeraiDatasetMain(n_features=self.n_features, dataset='train')
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        progress_bar = tqdm(total=len(data_loader)*n_epochs)
        losses = []
        loss_fn = nn.MSELoss()
        for epoch in range(n_epochs):
            for _, (X, y) in enumerate(data_loader):
                out = self.model.forward(X)
                loss = loss_fn(out, y)
                loss.backward()
                model.optimizer.step()
                model.optimizer.zero_grad()
                losses.append(loss)
                if len(losses) == 100:
                    losses.pop(0)
                progress_bar.set_description(f'Loss {np.mean(losses)}')
            if save_model:
                self.save_model()
    
    def run_inference(self, validation=False, load_model=True):
        if load_model:
            self.load_model()
        if validation:
            dataset = NumeraiDatasetMain(n_features=self.n_features, dataset='validation')
        else:
            dataset = NumeraiDatasetMain(n_features=self.n_features, dataset='live')
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        progress_bar = tqdm(total=len(data_loader), desc='Running Inference')
        losses = []
        loss_fn = nn.MSELoss()
        with T.no_grad:
            for _, (X, y) in enumerate(data_loader):
                out = self.model.forward(X)
                preds.append(out)
                if validation:
                    losses.append(loss_fn(out, y))

        if validation:
            print(f'Average Losses in valiation set; MSE: {np.mean(losses)}')
            dataset.data_handler.validation_df['preds_resnet_main'] = preds
            dataset.data_handler.validation_df[f'preds_resnet_main_neutral_riskiest_{self.n_features}'] =\
                    neutralize(
                    df=dataset.validation_df,
                    columns=['preds_resnet_main'],
                    neutralizers=dataset.features_list[-1],
                    proportion=1.0,
                    normalize=True,
                    n=self.n_features
                    )
            dataset.data_handler.validation_df['prediction'] = dataset.data_handler.validation_df[
                    f'preds_{self.model_file}_neutral_riskiest_{self.n}'
                    ].rank(pct=True)
            validation_sample_preds = pd.read_parquet('data_files/validation_example_preds.parquet')
            self.data_handler.validation_df['example_preds'] = validation_sample_preds['prediction']
            print('...Saving Validation Predictions...')
            self.data_handler.validation_df['prediction'].to_csv(
                    f'../predictions/validation_predictions_{dataset.handler.current_round}.csv'
                    )
            return
        dataset.data_handler.live_df[f'preds_resnet_main_neutral_riskiest_{self.n_features}'] = neutralize(
                    df=dataset.live_df,
                    columns=['preds_resnet_main'],
                    neutralizers=dataset.features_list[-1],
                    proportion=1.0,
                    normalize=True,
                    n=self.n_features
                    )
         
        dataset.data_handler.live_df['prediction'] = self.data_handler.live_df[
                f'preds_resnet_main_neutral_riskiest_{self.n_features}'
                ].rank(pct=True)
        print('...Saving Tournament Predictions...')
        if save_predictions:
            self.data_handler.live_df['prediction'].to_csv(
                    f'../predictions/tournament_predictions_{dataset.data_handler.current_round}.csv'
                    )
        return preds


    def save_model(self, filename='trained_models/resnet_main.pt'):
        print('...Saving Model...')
        T.save(self.model.state_dict(), filename)

    def load_model(self, filename='trained_models/resnet_main.pt'):
        print('...Loading Model...')
        self.model = self.model.load_state_dict(T.load(filename))



