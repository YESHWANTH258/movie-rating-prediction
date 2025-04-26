import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    def __init__(self):
        pass
    
    def calculate_director_success_rate(self, df):
        df = df.copy()
        df = df.sort_values('release_date')
        
        director_stats = df.groupby('director_encoded').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        director_stats.columns = ['director_encoded', 'director_avg_rating', 'director_movie_count']
        
        df = df.merge(director_stats, on='director_encoded', how='left')
        
        return df
    
    def calculate_genre_features(self, df):
        df = df.copy()
        
        genre_popularity = df.groupby('genres_encoded')['rating'].agg(['mean', 'count']).reset_index()
        genre_popularity.columns = ['genres_encoded', 'genre_avg_rating', 'genre_movie_count']
        
        df = df.merge(genre_popularity, on='genres_encoded', how='left')
        
        df['genre_popularity_score'] = df['genre_movie_count'] / df['genre_movie_count'].max()
        
        return df
    
    def calculate_temporal_features(self, df):
        df = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['release_date']):
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        df['release_day'] = df['release_date'].dt.day
        df['release_dayofweek'] = df['release_date'].dt.dayofweek
        df['release_quarter'] = df['release_date'].dt.quarter
        
        df = df.fillna(method='ffill')
        
        return df
    
    def calculate_similar_movie_features(self, df):
        df = df.copy()
        
        yearly_ratings = df.groupby('release_year')['rating'].mean().reset_index()
        yearly_ratings.columns = ['release_year', 'yearly_avg_rating']
        
        df = df.merge(yearly_ratings, on='release_year', how='left')
        
        return df
    
    def engineer_features(self, df):
        df = self.calculate_director_success_rate(df)
        df = self.calculate_genre_features(df)
        df = self.calculate_temporal_features(df)
        df = self.calculate_similar_movie_features(df)
        
        return df 