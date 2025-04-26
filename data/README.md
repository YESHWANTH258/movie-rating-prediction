# Data Requirements

This project requires two datasets:

1. `movies.csv` - Main movie dataset
2. `IMDb Movies India.csv` - Additional Indian movie dataset

## How to Get the Data

1. Download the datasets from the following sources:
   - `movies.csv`: [Kaggle Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
   - `IMDb Movies India.csv`: [IMDb Indian Movies Dataset](https://www.kaggle.com/datasets/ashirwadsangwan/imdb-indian-movies-dataset)

2. Place the downloaded files in this directory:
   ```
   data/
   ├── movies.csv
   └── IMDb Movies India.csv
   ```

## Data Description

### movies.csv
- Contains information about movies including:
  - Title
  - Release date
  - Genres
  - Ratings
  - Budget
  - Revenue

### IMDb Movies India.csv
- Contains information about Indian movies including:
  - Title
  - Year
  - Duration
  - Genre
  - Rating
  - Director
  - Cast

## Data Processing

The data will be automatically processed when you run the main script:
```bash
python src/main.py
```

The processed data will be saved in the `data/processed/` directory. 