import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
import logging
import os
from navisight.pipeline.feature_registry import REGISTRY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MaritimeFeatureScaler:
    def __init__(self, save_dir: str = "models/scalers"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.scaler = RobustScaler()
        
        # Sourced directly from registry
        self.continuous_cols = REGISTRY.continuous_scaled

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """FITS ONLY ON TRAINING SPLIT."""
        logging.info("Fitting RobustScaler on continuous physics features...")
        df_scaled = df.copy()
        
        scaled_values = self.scaler.fit_transform(df_scaled[self.continuous_cols])
        df_scaled[self.continuous_cols] = scaled_values
        
        scaler_path = os.path.join(self.save_dir, 'robust_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        return df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms Validation/Test splits without leaking statistics."""
        logging.info("Transforming features using pre-fitted scaler...")
        scaler_path = os.path.join(self.save_dir, 'robust_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        
        df_scaled = df.copy()
        scaled_values = self.scaler.transform(df_scaled[self.continuous_cols])
        df_scaled[self.continuous_cols] = scaled_values
        
        return df_scaled