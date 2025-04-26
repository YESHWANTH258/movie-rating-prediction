import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import chardet

class DataPreprocessor:
    def __init__(self):
        self.target_encoder = TargetEncoder()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_data(self, file_path):
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
            
        try:
            df = pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read the file with any of the attempted encodings")
                
        df = df.rename(columns={
            'Name': 'title',
            'Director': 'director',
            'Genre': 'genres',
            'Year': 'release_date',
            'Rating': 'rating'
        })
        return df
    
    def clean_data(self, df):
        df = df.copy()
        df = df.drop_duplicates()
        df.loc[:, 'release_date'] = pd.to_datetime(df['release_date'], format='%Y', errors='coerce')
        df.loc[:, 'rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        if 'Duration' in df.columns:
            df.loc[:, 'Duration'] = df['Duration'].str.extract('(\d+)').astype(float)
        
        if 'Votes' in df.columns:
            df.loc[:, 'Votes'] = df['Votes'].astype(str).str.replace(',', '')
            df.loc[:, 'Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
        
        return df
    
    def handle_missing_values(self, df):
        df = df.copy()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        df.loc[:, numerical_cols] = self.imputer.fit_transform(df[numerical_cols])
        
        for col in categorical_cols:
            mode_values = df[col].mode()
            if not mode_values.empty:
                df.loc[:, col] = df[col].fillna(mode_values.iloc[0])
            else:
                df.loc[:, col] = df[col].fillna('Unknown')
        
        return df
    
    def encode_categorical_variables(self, df, target_column):
        df = df.copy()
        categorical_cols = ['director', 'genres', 'Actor 1', 'Actor 2', 'Actor 3']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            df.loc[:, f'{col}_encoded'] = self.target_encoder.fit_transform(df[col], df[target_column])
            
        df = df.drop(columns=categorical_cols)
        
        return df
    
    def scale_features(self, df, target_column):
        df = df.copy()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = numerical_cols.drop(target_column)
        
        df.loc[:, numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def preprocess(self, file_path, target_column='rating'):
        df = self.load_data(file_path)
        df = self.clean_data(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical_variables(df, target_column)
        df = self.scale_features(df, target_column)
        
        return df 