import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from numerapi import NumerAPI
from Numerai.data_handling import DataHandler
import joblib
import optuna
import gc
from utils import validation_metrics, neutralize


class BasicLGBM:
    def __init__(self, feature_set='medium', n=50, secondary_targets=None, 
                 model_file='lgb.pkl'):
        self.handler = DataHandler(feature_set=feature_set, 
                                   secondary_targets=secondary_targets)
        self.feature_corrs = self.get_corrs()
        self.top_n_feats = self.handler.get_n_most_corr(self.feature_corrs, n)
        self.n = n
        self.model_file = model_file
        self.params = {
                'n_estimators': 2000,
                'learning_rate': 0.01,
                'max_depth': 5,
                'num_leaves': 2 ** 5,
                'colsample_bytree': 0.1
                }

    def get_corrs(self):
        feature_corrs = self.handler.train_df.groupby('era').apply(
                lambda era: era[self.handler.features]\
                        .corrwith(era[self.handler.target])
                )
        gc.collect()
        return feature_corrs

    def search_hyperparams(self):
        params = {
                'n_estimators': trial.suggest_int('num_estimators', 2000, 12000),
                'learning_rate': trial.suggest_uniform('learning_rate', 0, 0.2),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'num_leaves': 2 ** trial.suggest_int('num_leaves', 4, 8),
                'colsample_bytree': trial.suggest_int('colsample_bytree', 0.05, 0.2)
                }
        model = LGBMClassifier(**params)
        model.fit(
                self.handler.train_df.filter(like='feature_', axis='columns'),
                self.handler.train_df[self.handler.target]
                )
        score = model.score(
                self.handler.validation_df.filter(like='feature_', axis='columns'),
                self.handler.validation_df_df[self.handler.target]
                )
        return score

    def get_optimal_model(self, n_rounds=10, save_model=True):
        studyLGBM = optuna.create_study(direction='maximize')
        studyLGBM.sampler = optuna.samplers.TPESampler()
        studyLGBM.optimize(self.search_hyperparams, n_trials=n_trials)

        trial = studyLGBM.best_trial
        params_best = dict(trial.params.items())
        
        model = LGBMClassifier(**params_best)
        model.fit(
                self.handler.train_df.filter(like='feature_', axis='columns'),
                self.handler.train_df[self.handler.target]
                )
        if save_model:
            joblib.dump(model, f'../trained_models/{self.model_file}')
        return model


    def train(self, save_model=True):
        model = LGBMRegressor(**self.params)
        print('...Training LGBM Model...')
        model.fit(
                self.handler.train_df.filter(like='feature_', axis='columns'),
                self.handler.train_df[self.handler.target]
                )
        if save_model:
            print('...Saving Model...')
            joblib.dump(model, '../trained_models/lgb.pkl')
        gc.collect()
        return model
    
    def validate(self, model, save_predictions=True):
        print('...Making Validation Predictions...')
        self.handler.validation_df.loc[:, f'preds_{self.model_file}'] = model.predict(
                self.handler.live_df.loc[:, model.booster_.feature_name()]
                )
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
        validation_sample_preds = pd.read_parquet(f'{self.handler.version}/
                validation_example_preds.parquet')
        validation_sample_preds['example_preds'] = self.handler.validation_df['prediction']
        validation_stats = validation_metrics(
                validation_data, 
                [f'preds_{self.model_file}_neutral_riskiest_{n}', f'preds_{model_file}'], 
                example_col='example_preds',
                fast_mode=True, 
                target_col=self.handler.target
                )
        print(validation_stats[["mean", "sharpe"]].to_markdown())
        print('...Saving Validation Predictions...')
        if save_predictions:
            validation_sample_preds['example_preds'].to_csv(
                    f'../predictions/validation_predictions_{self.handler.current_round}.csv'
                    )

    def make_predictions(self, model, save_predictions=True):
        nans_per_col = self.handler.live_df[self.handler.live_df['data_type']\
                        == 'live'][self.handler.features].isna().sum()
        if nans_per_col.any():
            self.handler.live_df.loc[:, self.handler.features] = \
                         self.handler.live_df.loc[:, self.handler.features].fillna(0.5)
        assert set(model.booster_.feature_name()) == set(self.handler.features), 
               'There are new features available. Retrain this model before making any 
                further submissions.' 
        print('...Making Live Predictions...')
        self.handler.live_df.loc[:, f'preds_{self.model_file}'] = model.predict(
                self.handler.live_df.loc[:, model.booster_.feature_name()]
                )
        gc.collect()
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
