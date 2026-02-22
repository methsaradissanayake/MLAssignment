import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, data, target_col):
        self.df = data
        self.target_col = target_col
        self.features = []
        self.brand_mapping = {}
        self.model_mapping = {}

    def extract_mappings(self):
        """Extracts dictionaries connecting string Names to Encoded Values."""
        if 'Brand' in self.df.columns and 'Brand_Encoded' in self.df.columns:
            brand_map = self.df.drop_duplicates('Brand_Encoded').set_index('Brand_Encoded')['Brand'].to_dict()
            self.brand_mapping = {str(k): v for k, v in brand_map.items()}
            
        if 'Model' in self.df.columns and 'Model_Encoded' in self.df.columns:
            model_map = self.df.drop_duplicates('Model_Encoded').set_index('Model_Encoded')['Model'].to_dict()
            self.model_mapping = {str(k): v for k, v in model_map.items()}
            
        return self.brand_mapping, self.model_mapping

    def select_features(self, ignore_cols=None):
        """Drops unneeded text columns, retains encoded/numeric features."""
        if ignore_cols is None:
            ignore_cols = ['Title', 'Description', 'PublishedDate', 'Link', 'ImageURL', 'Location', 'Brand', 'Model', 'Location_Clean', 'Price_Normalized', 'Mileage_Normalized']
            
        existing_drops = [c for c in ignore_cols if c in self.df.columns]
        self.df = self.df.drop(columns=existing_drops)
        
        # Fill remaining missing numeric data with 0
        self.df = self.df.fillna(0)
        
        self.features = [c for c in self.df.columns if c != self.target_col]
        print(f"Selected {len(self.features)} features for training.")
        return self.df

    def split_and_save(self, output_dir="data", test_size=0.15, val_size=0.15):
        """Splits into Train/Val/Test and saves to directory alongside metadata."""
        X = self.df[self.features]
        y = self.df[self.target_col]
        
        # Math for splitting a remaining set
        temp_size = test_size + val_size
        val_ratio = val_size / temp_size
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=temp_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42)
        
        print(f"Data Splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        metadata = {
            'features': self.features,
            'target': self.target_col,
            'brand_mapping': self.brand_mapping,
            'model_mapping': self.model_mapping
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
            
        print(f"Preprocessed artifacts saved to {output_dir}/")
        return output_dir
