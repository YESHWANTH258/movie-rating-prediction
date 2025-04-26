import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class MovieRatingPredictor:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    
    def prepare_data(self, df, target_column):
        df = df.copy()
        
        columns_to_drop = ['title', 'release_date']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        if 'Duration' in df.columns:
            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        cv_scores = cross_val_score(
            self.model, X_test, y_test, cv=5, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_RMSE_mean': cv_rmse.mean(),
            'CV_RMSE_std': cv_rmse.std()
        }
    
    def plot_feature_importance(self, X_train):
        importance = self.model.feature_importances_
        feature_names = X_train.columns
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df.head(10))
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
    
    def load_model(self, file_path):
        self.model = joblib.load(file_path)
        return self.model
    
    def predict(self, X):
        return self.model.predict(X) 