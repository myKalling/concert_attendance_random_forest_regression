# Concert Attendance Prediction System

## Overview

A comprehensive machine learning system for predicting concert attendance using Python, scikit-learn, and external data sources. This project demonstrates the complete ML workflow: data collection, validation, cleaning, feature engineering, external data integration, and predictive modeling using Object-Oriented Python design.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy .env example and fill with your own API credentials
copy .env.example .env
```

Then edit `.env` with your private keys:

```ini
SPOTIPY_CLIENT_ID=
SPOTIPY_CLIENT_SECRET=
TICKETMASTER_API_KEY=
EVENTBRITE_TOKEN=
CENSUS_API_KEY=
```

Run the full pipeline:

```bash
python pipeline_script.py --input concerts_raw.csv --output-dir ./results
```

Or quick mode without external API fetches:

```bash
python pipeline_script.py --quick
```

## Project Structure

**Core Modules:**
- `pipeline_script.py` - Main ML pipeline (ConcertModel, PipelineOrchestrator classes)
- `data_processor.py` - Data processing (DataValidator, DataCleaner, FeatureEngineer classes)
- `data_fetchers_oop.py` - External data fetchers (SpotifyArtistFetcher, WeatherFetcher, etc.)
- `dataset_merger.py` - Dataset merging (DatasetMerger class)

**Data Files:**
- `concerts_raw.csv` - Raw input data
- `concerts_processed.csv` - After cleaning & engineering
- `concerts_merged.csv` - Final training dataset

## Key Features

### Object-Oriented Design
- DataValidator: Schema validation, missing value detection
- DataCleaner: Date parsing, normalization, imputation
- FeatureEngineer: Temporal and price-based features
- SpotifyArtistFetcher: Fetch from Spotify API
- WeatherFetcher: Fetch historical weather (Open-Meteo, no key needed)
- DatasetMerger: Merge multiple data sources
- ConcertModel: ML model with train/test split, cross-validation
- PipelineOrchestrator: Coordinate complete workflow

### Data Processing Pipeline
1. **Validation**: Schema check, duplicate detection, missing value analysis
2. **Cleaning**: Date parsing, city normalization, imputation, deduplication
3. **Feature Engineering**: Temporal features (day_of_week, month), price-based features, target variable
4. **External Data**: Fetch Spotify, weather, Ticketmaster, Eventbrite, Census demographics
5. **Merging**: Intelligent join on genre, city, date
6. **Modeling**: Train/test split, preprocessing, RandomForest/GradientBoosting, cross-validation

### Machine Learning
- Train/test split (80/20)
- Preprocessing: StandardScaler + OneHotEncoder
- Random Forest with 200 trees
- 5-fold cross-validation
- Evaluation: R², RMSE, MAE

## Installation

```bash
pip install pandas numpy scikit-learn requests
```

## Usage

```bash
# Full pipeline with external data
python pipeline_script.py --input concerts_raw.csv --output-dir ./results

# Quick mode (skip external data)
python pipeline_script.py --quick

# Custom input/output
python pipeline_script.py -i data.csv -o ./output
```

## Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| event_id | Numeric | Concert ID |
| event_date | Date | Event date |
| city | Categorical | Venue city |
| venue_capacity | Numeric | Max capacity |
| artist_popularity | Numeric | Artist score (0-100) |
| ticket_price | Numeric | Price (USD) |
| competing_events | Numeric | Other events same date/city |
| genre | Categorical | Music genre |
| attendance | Numeric | Tickets sold |
| **attendance_rate** | **Numeric** | **Target = attendance / capacity** |
| day_of_week | Numeric | 0=Mon, 6=Sun |
| is_weekend | Numeric | 1 if Fri-Sun |
| price_per_capacity | Numeric | Price / capacity |
| popularity_price | Numeric | Popularity × price |

## Output Files

- `concerts_processed.csv` - Cleaned & engineered
- `spotify.csv` - Artist data
- `weather.csv` - Historical weather
- `concerts_merged.csv` - Final dataset
- `model_summary.txt` - Evaluation results

## Extending the System

Add custom data source:

```python
from data_fetchers_oop import DataFetcher
import pandas as pd

class CustomFetcher(DataFetcher):
    def fetch(self) -> pd.DataFrame:
        # Your fetching logic
        return pd.DataFrame(data)
```

## References

- [scikit-learn](https://scikit-learn.org/)
- [Spotify API](https://developer.spotify.com/documentation/web-api/)
- [Open-Meteo API](https://open-meteo.com/)
- [pandas](https://pandas.pydata.org/)

## Author

Applied Machine Learning Final Project - Concert Attendance Prediction

## License

Educational - use freely for learning purposes
