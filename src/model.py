from re import sub
from typing import List, Union, Optional
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso
)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, 
    AdaBoostRegressor,
    RandomForestRegressor
)
import xgboost as xgb
from xgboost import Booster as XGBoostRegressor


class AllModelsEnum():
    
    # Linear regression models
    linear_regressor = LinearRegression
    logistic_regressor = LogisticRegression
    ridge_regressor = Ridge
    lasso_regressor = Lasso
    
    # Neural network models
    # TODO: Add transformers and LSTMs.
    mlp_regressor = MLPRegressor
    
    # Ensemble models
    gradient_boosting_regressor = GradientBoostingRegressor
    ada_boost_regressor = AdaBoostRegressor
    random_forest_regressor = RandomForestRegressor
    xgboost_regressor = XGBoostRegressor
    
    @staticmethod
    def get_all_model_strings() -> List[str]:
        return [x for x in dir(AllModelsEnum) if 'regressor' in x]


class Model(BaseEstimator):
    
    def __init__(
        self, 
        model: Optional[Union[BaseEstimator, xgb.Booster]] = LinearRegression
    ) -> None:
        self.model = model()
        self.fitted_model = None

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.fitted_model = self.model.fit(X, y)
    
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.fitted_model.predict(X)
