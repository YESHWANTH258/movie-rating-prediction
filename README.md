# Movie Rating Prediction System

## Task Objectives
This project aims to develop a machine learning system that predicts movie ratings based on various attributes. The system:
- Analyzes movie attributes like director, genre, and release date
- Processes and transforms raw movie data
- Creates predictive features from movie metadata
- Trains and evaluates a machine learning model
- Provides accurate rating predictions for new movies

## Project Structure
```
Movie_Rating/
├── data/                  # Data directory
│   └── raw/              # Raw data files
├── src/                  # Source code
│   ├── data_preprocessing.py    # Data cleaning and preparation
│   ├── feature_engineering.py   # Feature creation
│   ├── model_training.py        # Model implementation
│   └── main.py                 # Main execution script
├── models/              # Trained models
├── notebooks/           # Analysis notebooks
├── tests/              # Test files
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie-rating-prediction.git
   cd movie-rating-prediction
   ```

2. Create and activate virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project
1. Place your movie dataset in `data/raw/` directory
2. Execute the main script:
   ```bash
   python src/main.py
   ```

## Expected Output
- Preprocessed dataset
- Trained model file
- Model evaluation metrics
- Feature importance visualization

## Model Performance
- Mean Absolute Error: 0.2881
- Root Mean Squared Error: 0.5298
- R-squared Score: 0.7262
- Cross-Validation RMSE: 0.5636 ± 0.0097

## Key Features
- Automated data preprocessing
- Advanced feature engineering
- XGBoost-based prediction model
- Comprehensive model evaluation
- Feature importance analysis

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License. 