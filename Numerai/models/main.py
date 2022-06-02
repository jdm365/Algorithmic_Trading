import pandas as pd
import numpy as np
from basic_lgbm import BasicLGBM


if __name__ == '__main__':
    model = BasicLGBM(feature_set='medium', n=75, model_file='lgb.pkl')
    best_model = model.get_optimal_model(n_rounds=10)
    model.validate(best_model, save_predictions=True)
    model.make_predictions(best_model, save_predictions=True)
