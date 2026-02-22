import os
import json
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

class XGBModelBuilder:
    def __init__(self, data_dir, models_dir="models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.model = None

    def _load_data(self):
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(self.data_dir, 'val.csv'))
        
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        target = metadata['target']
        features = metadata['features']
        
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]
        return X_train, y_train, X_val, y_val

    def tune_and_train(self, n_iter=10):
        print("Initializing Hyperparameter Tuning via RandomizedSearchCV...")
        X_train, y_train, X_val, y_val = self._load_data()

        param_distributions = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5, 7]
        }
        
        base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='neg_root_mean_squared_error',
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        print(f"Best params: {random_search.best_params_}")
        
        print("Training finalized model with early stopping on validation set...")
        best_params = random_search.best_params_
        best_params.update({'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 50})
        
        self.model = xgb.XGBRegressor(**best_params)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        return self.model

    def save_model(self, filename='xgb_model.pkl'):
        if not self.model:
            raise ValueError("Model is not trained. Call tune_and_train() first.")
            
        os.makedirs(self.models_dir, exist_ok=True)
        path = os.path.join(self.models_dir, filename)
        joblib.dump(self.model, path)
        print(f"Model saved securely to {path}")
