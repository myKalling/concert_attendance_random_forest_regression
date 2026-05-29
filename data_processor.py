"""
Data Processor: Validation, Cleaning, and Feature Engineering
Demonstrates OOP with DataValidator, DataCleaner, and FeatureEngineer classes
"""

import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and schema completeness."""

    def __init__(self, required_columns: Optional[List[str]] = None):
        self.required_columns = required_columns or []
        self.report = {}

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check if required columns exist."""
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            logger.error(f"Missing required columns: {missing}")
            self.report['missing_columns'] = list(missing)
            return False
        return True

    def check_quality(self, df: pd.DataFrame, missing_threshold: float = 0.5) -> Dict:
        """Check for missing values, duplicates, and numeric consistency."""
        missing = (df.isnull().sum() / len(df)).round(4)
        high_missing = missing[missing > missing_threshold]
        dups = df.duplicated().sum()

        issues = {}
        for col in ['venue_capacity', 'attendance', 'ticket_price']:
            if col in df.columns:
                neg = (df[col] < 0).sum()
                if neg > 0:
                    issues[col] = f"{neg} negative values"

        stats = {
            'total_missing': int(df.isnull().sum().sum()),
            'duplicates': int(dups),
            'high_missing_cols': high_missing.to_dict(),
            'numeric_issues': issues
        }

        if high_missing.size > 0:
            logger.warning(f"High missing values: {high_missing.to_dict()}")
        if dups > 0:
            logger.warning(f"Found {dups} duplicate rows")
        if issues:
            logger.warning(f"Numeric issues: {issues}")

        self.report.update(stats)
        return stats


class DataCleaner:
    """Cleans and normalizes data."""

    def __init__(self):
        self.log = []

    def _log(self, msg: str):
        self.log.append(msg)
        logger.info(msg)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning operations."""
        df = df.copy()

        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
            self._log("Parsed event_date")

        if 'city' in df.columns:
            mapping = {
                'la': 'Los Angeles', 'l.a.': 'Los Angeles', 'la, ca': 'Los Angeles',
                'nyc': 'New York', 'ny': 'New York', 'new york city': 'New York',
                'new york, ny': 'New York', 'chgo': 'Chicago', 'chi': 'Chicago',
                'chicago, il': 'Chicago', 'sf': 'San Francisco', 'san francisco, ca': 'San Francisco',
            }
            before = df['city'].nunique()
            df['city'] = df['city'].astype(str).str.strip().str.lower()
            df['city'] = df['city'].map(mapping).fillna(df['city']).str.title()
            after = df['city'].nunique()
            self._log(f"Normalized cities: {before} → {after} unique")

        numeric_cols = ['ticket_price', 'artist_popularity', 'competing_events']
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                self._log(f"Imputed {col} with median: {fill_val:.2f}")

        if 'attendance' in df.columns and 'venue_capacity' in df.columns:
            over = (df['attendance'] > df['venue_capacity']).sum()
            if over > 0:
                df.loc[df['attendance'] > df['venue_capacity'], 'attendance'] = \
                    df.loc[df['attendance'] > df['venue_capacity'], 'venue_capacity']
                self._log(f"Capped {over} over-capacity attendance values")

        if 'genre' in df.columns:
            before = df['genre'].nunique()
            top_genres = df['genre'].value_counts().head(6).index
            df['genre'] = df['genre'].where(df['genre'].isin(top_genres), 'Other')
            after = df['genre'].nunique()
            self._log(f"Filtered genres: {before} → {after} categories")

        before = len(df)
        df = df.drop_duplicates(subset=['event_id'], keep='first')
        removed = before - len(df)
        if removed > 0:
            self._log(f"Removed {removed} duplicate rows")

        return df


class FeatureEngineer:
    """Creates derived features for modeling."""

    def __init__(self):
        self.features_created = []

    def engineer(self, df):
        df = df.copy()

        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
            df['day_of_week'] = df['event_date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['month'] = df['event_date'].dt.month
            df['quarter'] = df['event_date'].dt.quarter
            df['day_of_month'] = df['event_date'].dt.day
            self.features_created.extend(['day_of_week', 'is_weekend', 'month', 'quarter', 'day_of_month'])
            logger.info("Created temporal features")

        if 'ticket_price' in df.columns and 'venue_capacity' in df.columns:
            df['price_per_capacity'] = df['ticket_price'] / (df['venue_capacity'] + 1e-6)
            self.features_created.append('price_per_capacity')

        if 'ticket_price' in df.columns and 'artist_popularity' in df.columns:
            df['popularity_price'] = df['ticket_price'] * df['artist_popularity']
            self.features_created.append('popularity_price')

        if 'attendance' in df.columns and 'venue_capacity' in df.columns:
            df['attendance_rate'] = df['attendance'] / (df['venue_capacity'] + 1e-6)
            df['attendance_rate'] = df['attendance_rate'].clip(0, 1)
            self.features_created.append('attendance_rate')
            logger.info("Created attendance_rate target variable")

        return df


def prepare_data(input_csv: str, output_csv: str = 'concerts_processed.csv') -> pd.DataFrame:
    logger.info(f"Starting data preparation from {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    validator = DataValidator([
        'event_id', 'event_date', 'city', 'venue_capacity',
        'artist_popularity', 'ticket_price', 'attendance', 'genre'
    ])
    validator.validate_schema(df)
    validator.check_quality(df)

    cleaner = DataCleaner()
    df = cleaner.clean(df)

    engineer = FeatureEngineer()
    df = engineer.engineer(df)

    df.to_csv(output_csv, index=False)
    logger.info(f"Processed data saved to {output_csv}: {df.shape}")
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    try:
        df = prepare_data('concerts_raw.csv')
        print(f"\n✓ Data processing complete: {df.shape}")
    except Exception as e:
        print(f"Error: {e}")
