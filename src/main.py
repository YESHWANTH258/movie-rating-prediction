import os
import pandas as pd
import traceback
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import MovieRatingPredictor

def check_data_files():
    """Check if required data files exist and provide guidance if they don't."""
    required_files = {
        'data/movies.csv': 'Main movie dataset',
        'data/IMDb Movies India.csv': 'Indian movie dataset'
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append((file_path, description))
    
    if missing_files:
        print("\nError: Required data files are missing!")
        print("\nPlease download the following files and place them in the data/ directory:")
        for file_path, description in missing_files:
            print(f"\n- {file_path} ({description})")
        print("\nInstructions for downloading the data files can be found in data/README.md")
        return False
    return True

def main():
    try:
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Check for required data files
        if not check_data_files():
            return
        
        # Initialize components
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()
        predictor = MovieRatingPredictor()
        
        # Define file paths
        data_path = 'data/movies.csv'
        model_path = 'models/movie_rating_predictor.joblib'
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        try:
            df = preprocessor.preprocess(data_path, target_column='rating')
            print(f"Data loaded successfully. Shape: {df.shape}")
            print("Columns in dataset:", list(df.columns))
        except Exception as e:
            print("Error during preprocessing:")
            print(traceback.format_exc())
            raise
        
        # Engineer features
        print("\nEngineering features...")
        try:
            df = feature_engineer.engineer_features(df)
            print("Feature engineering completed successfully")
        except Exception as e:
            print("Error during feature engineering:")
            print(traceback.format_exc())
            raise
        
        # Prepare data for model training
        print("\nPreparing data for model training...")
        try:
            X_train, X_test, y_train, y_test = predictor.prepare_data(df, 'rating')
            print(f"Training data shape: {X_train.shape}")
            print(f"Test data shape: {X_test.shape}")
        except Exception as e:
            print("Error during data preparation:")
            print(traceback.format_exc())
            raise
        
        # Train model
        print("\nTraining model...")
        try:
            predictor.train_model(X_train, y_train)
            print("Model training completed successfully")
        except Exception as e:
            print("Error during model training:")
            print(traceback.format_exc())
            raise
        
        # Evaluate model
        print("\nEvaluating model...")
        try:
            metrics = predictor.evaluate_model(X_test, y_test)
            
            # Print evaluation metrics
            print("\nModel Evaluation Metrics:")
            print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
            print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
            print(f"R-squared Score: {metrics['R2']:.4f}")
            print(f"Cross-Validation RMSE: {metrics['CV_RMSE_mean']:.4f} ± {metrics['CV_RMSE_std']:.4f}")
        except Exception as e:
            print("Error during model evaluation:")
            print(traceback.format_exc())
            raise
        
        # Plot feature importance
        print("\nGenerating feature importance plot...")
        try:
            predictor.plot_feature_importance(X_train)
            print("Feature importance plot generated successfully")
        except Exception as e:
            print("Error during feature importance plotting:")
            print(traceback.format_exc())
            raise
        
        # Save model
        print("\nSaving model...")
        try:
            predictor.save_model(model_path)
            print("Model saved successfully")
        except Exception as e:
            print("Error during model saving:")
            print(traceback.format_exc())
            raise
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nDetailed error information:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 