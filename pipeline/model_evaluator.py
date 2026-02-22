import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluator:
    def __init__(self, data_dir="data", models_dir="models", outputs_dir="outputs"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.outputs_dir = outputs_dir
        self.plots_dir = os.path.join(self.outputs_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Loaded artifacts
        self.model = None
        self.X_test = None
        self.y_test = None
        
    def _load_artifacts(self):
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        target = metadata['target']
        features = metadata['features']
        
        self.X_test = test_df[features]
        self.y_test = test_df[target]
        self.model = joblib.load(os.path.join(self.models_dir, 'xgb_model.pkl'))

    def evaluate_performance(self):
        """Calculates regression metrics vs Unseen Test set."""
        if self.model is None:
            self._load_artifacts()
            
        preds = self.model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)
        
        metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        
        with open(os.path.join(self.outputs_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
            
        pd.DataFrame([metrics]).to_csv(os.path.join(self.outputs_dir, 'metrics_table.csv'), index=False)
        
        print(f"Metrics: R2={r2:.4f} | RMSE={rmse:,.0f} | MAE={mae:,.0f}")
        
    def generate_plots(self):
        if self.model is None:
            self._load_artifacts()
            
        preds = self.model.predict(self.X_test)
        
        # 1. Prediction vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, preds, alpha=0.5, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Predicted vs Actual Prices')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'predicted_vs_actual.png'))
        plt.close()

    def generate_shap(self):
        print("Generating SHAP Explainability visuals...")
        if self.model is None:
            self._load_artifacts()
            
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X_test)
        
        # Feature Importance Profile
        plt.figure()
        shap.summary_plot(shap_values, self.X_test, plot_type="bar", show=False)
        plt.title("Absolute Average Feature Effect on Prediction")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'feature_importance_bar.png'))
        plt.close()
        
        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.title("SHAP Global Effect Mapping")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'shap_summary_plot.png'))
        plt.close()
