import pandas as pd
import numpy as np
import re

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.target_col = None

    def load_data(self):
        """Reads CSV into a Pandas DataFrame."""
        print(f"Loading data from {self.file_path}...")
        self.df = pd.read_csv(self.file_path)
        return self.df

    def identify_target(self, possible_names=None):
        """Finds the price target column based on naming conventions."""
        if possible_names is None:
            possible_names = ['price', 'amount']
            
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Default fast check
        if 'Price' in self.df.columns:
            self.target_col = 'Price'
        else:
            for col in self.df.columns:
                if any(name in col.lower() for name in possible_names):
                    self.target_col = col
                    break

        if not self.target_col:
            raise ValueError(f"Target column could not be identified using names: {possible_names}")
            
        print(f"Target column identified: '{self.target_col}'")
        return self.target_col

    def _clean_price(self, val):
        """Handles currency string anomalies (Rs, Lakh, Mn)."""
        if pd.isna(val):
            return np.nan
        
        val_str = str(val).lower().replace(',', '').replace(' ', '')
        val_str = val_str.replace('rs.', '').replace('rs', '')
        
        multiplier = 1.0
        if 'lakh' in val_str:
            multiplier = 100_000.0
            val_str = val_str.replace('lakh', '')
        elif 'mn' in val_str or 'million' in val_str:
            multiplier = 1_000_000.0
            val_str = val_str.replace('mn', '').replace('million', '')
            
        numeric_match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
        if numeric_match:
            try:
                return float(numeric_match.group()) * multiplier
            except:
                return np.nan
        return np.nan

    def clean_target(self):
        """Applies price cleaning and drops invalid rows."""
        if not self.target_col:
            self.identify_target()
            
        self.df[self.target_col] = self.df[self.target_col].apply(self._clean_price)
        
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=[self.target_col])
        print(f"Dropped {initial_len - len(self.df)} rows due to missing/invalid target values.")
        return self.df
