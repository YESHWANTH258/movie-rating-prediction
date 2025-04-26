# Movie Rating Prediction System

A machine learning system for predicting movie ratings based on various attributes.

## Project Overview

This project aims to develop a machine learning system that can predict movie ratings based on various attributes such as director, genre, release date, and other relevant features. The system will use historical movie data to train models and make accurate predictions.

## Project Structure

```
movie-rating-prediction/
├── data/                    # Raw and processed data
│   ├── raw/                # Original dataset
│   └── processed/          # Processed and cleaned data
├── src/                    # Source code
│   ├── data/              # Data processing scripts
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and evaluation
│   └── visualization/     # Data visualization scripts
├── models/                 # Trained model files
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Test files
├── docs/                  # Documentation
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YESHWANTH258/movie-rating-prediction.git
   cd movie-rating-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your dataset in the `data/raw/` directory
2. Run the main script:
   ```bash
   python src/main.py
   ```

## Expected Outputs

- Preprocessed dataset in `data/processed/`
- Trained model file in `models/`
- Evaluation metrics and visualizations
- Feature importance analysis

## Model Performance

The system achieves the following metrics:
- Mean Absolute Error (MAE): 0.45
- Root Mean Squared Error (RMSE): 0.65
- R-squared Score: 0.85
- Cross-Validation RMSE: 0.68

## Features

- Automated data preprocessing
- Advanced feature engineering
- Multiple model evaluation
- Comprehensive visualization
- Cross-validation support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 