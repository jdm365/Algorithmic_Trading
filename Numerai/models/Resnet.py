import pandas as pd
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('../')
from data_handling import DataHandler
import gc
from utils import validation_metrics, neutralize, get_corrs, Resnet


class BasicResnet(nn.Module):
    def __init__(self, lr, feature_set='all', n=50, batch_size=64, model_file='Resnet.pt'):
        super(BasicResnet, self).__init__()
        self.handler = DataHandler(
                feature_set=feature_set,
                secondary_targets=secondary_targets,
                get_dataloader=True,
                batch_size=batch_size
                )
        self.feature_corrs = get_corrs(
                df=self.handler.train_df, 
                features=self.handler.features, 
                target=self.handler.target
                )
        self.model_file = model_file
        self.n = n
        self.model = Resnet(
                lr=lr,
                input_dims=[34, 35],
                in_features=32,
                n_residual_blocks=5,
                output_features=1,
                kernel_sizes=[7, 5, 3, 3, 1],
                paddings=[3, 2, 1, 1, 0],
                strides=[1, 1, 1, 1, 1],
                is_regressor=True
                )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def train(self, n_epochs, save_model=True):
        progress_bar = tqdm(total=n_epochs*len(self.handler.train_loader))
        losses = []
        loss_fn = nn.MSELoss()
        for epoch in range(n_epochs):
            for idx, (X, y) in enumerate(self.handler.train_loader):
                out = self.model.forward(X)
                loss = loss_fn(out, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss)
                if len(losses) > 50:
                    losses.pop(0)
                progress_bar.set_description(f'Epoch {epoch+1} \t Loss: {np.mean(losses)}')
                progress_bar.update(1)
                if save_model and idx % 2500:
                    T.save(self.model.state_dict(), f'trained_models/{self.model_file}')
            if save_model:
                T.save(self.model.state_dict(), f'trained_models/{self.model_file}')
        progress_bar.close()
        return self.model
        

    def validate(self, model=None, save_predictions=True):
        if model is None:
            model = self.model.load_state_dict(T.load(f'trained_models/{self.model_file}'))
        preds = []
        with T.no_grad():
            for idx, (X, _) in enumerate(tqdm(self.handler.validation_loader, 
                                         desc='Making Validation Predictions')):
                out = model.forward(X).detach().cpu().numpy().tolist()
                preds += out

        self.handler.validation_df.loc[:, f'preds_{self.model_file}'] = preds 
        self.handler.validation_df[f'preds_{self.model_file}_neutral_riskiest_{n}'] = neutralize(
                df=self.handler.validation_df,
                columns=[f'preds_{self.model_file}'],
                neutralizers=riskiest_features,
                proportion=1.0,
                normalize=True,
                n=self.n
                )
        self.handler.validation_df['prediction'] = self.handler.validation_df[
                f'preds_{self.model_file}_neutral_riskiest_{n}'
                ].rank(pct=True)
        validation_sample_preds = pd.read_parquet(f'{self.handler.version}/\
                validation_example_preds.parquet')
        validation_sample_preds['example_preds'] = self.handler.validation_df['predictions']
        validation_stats = validation_metrics(
                validation_data,
                [f'preds_{self.model_file}_neutral_riskiest_{n}', f'preds_{model_file}'],
                example_col='example_preds',
                fast_mode=True,
                target_col=self.handler.target
                )
        print(validation_stats[['mean', 'sharpe']].to_markdown())
        print('...Saving Validation Predictions...')
        if save_predictions:
            valiation_sample_preds['example_preds'].to_csv(
                f'../predictions/validation_predictions_{self.handler.current_round}.csv'
                )

    def make_predictions(self, model=None, save_predicitions=True):
        nans_per_col = self.handler.live_df[self.handler.live_df['data_type']\
                        == 'live'][self.handler.features].isna().sum()
        if nans_per_col.any():
            self.handler.live_df.loc[:, self.handler.features] = \
                         self.handler.live_df.loc[:, self.handler.features].fillna(0.5)

        if model is None:
            model = self.model.load_state_dict(T.load(f'trained_models/{self.model_file}'))

        preds = []
        with T.no_grad():
            for idx, (X, _) in enumerate(tqdm(self.handler.validation_loader, 
                                         desc='Making Live Predictions')):
                out = model.forward(X).detach().cpu().numpy().tolist()
                preds += out

        self.handler.live_df.loc[:, f'preds_{self.model_file}'] = preds 
        self.handler.live_df[f'preds_{self.model_file}_neutral_riskiest_{n}'] = neutralize(
                df=self.handler.live_df,
                columns=[f'preds_{self.model_file}'],
                neutralizers=riskiest_features,
                proportion=1.0,
                normalize=True,
                n=self.n
                )
        self.handler.live_df['prediction'] = self.handler.live_df[
                f'preds_{self.model_file}_neutral_riskiest_{n}'
                ].rank(pct=True)
        print('...Saving Tournament Predictions...')
        if save_predictions:
            self.handler.live_df['prediction'].to_csv(
                    f'../predictions/tournament_predictions_{self.handler.current_round}.csv'
                    )
