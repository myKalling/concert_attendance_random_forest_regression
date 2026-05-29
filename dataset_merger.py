"""
Dataset Merger: Intelligently merge multiple data sources
Demonstrates composition and encapsulation
"""

import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class DatasetMerger:
    def __init__(self):
        self.df = None
        self.merge_log = []

    def _log(self, msg: str):
        self.merge_log.append(msg)
        logger.info(msg)

    def load_primary(self, concerts_csv: str) -> pd.DataFrame:
        self.df = pd.read_csv(concerts_csv)
        self._log(f"Loaded primary dataset: {self.df.shape}")
        return self.df

    def merge_spotify(self, spotify_df: pd.DataFrame) -> None:
        if self.df is None:
            raise ValueError("Primary dataset not loaded")
        if 'genre' not in self.df.columns or 'genre' not in spotify_df.columns:
            logger.warning("Skipping Spotify merge: genre not available")
            return
        before = self.df.shape[1]
        self.df = self.df.merge(spotify_df, on='genre', how='left')
        self._log(f"Merged Spotify genre data: +{self.df.shape[1] - before} columns")

    def merge_weather(self, weather_df: pd.DataFrame) -> None:
        if self.df is None:
            raise ValueError("Primary dataset not loaded")
        if {'city', 'event_date'}.issubset(weather_df.columns):
            weather_df['event_date'] = pd.to_datetime(weather_df['event_date'], errors='coerce')
            self.df['event_date'] = pd.to_datetime(self.df['event_date'], errors='coerce')
            before = self.df.shape[1]
            self.df = self.df.merge(weather_df, on=['city', 'event_date'], how='left')
            self._log(f"Merged weather data: +{self.df.shape[1] - before} columns")
        else:
            logger.warning("Skipping weather merge: missing required columns")

    def merge_ticketmaster(self, ticketmaster_df: pd.DataFrame) -> None:
        if self.df is None:
            raise ValueError("Primary dataset not loaded")
        if {'city', 'event_date'}.issubset(ticketmaster_df.columns):
            ticketmaster_df['event_date'] = pd.to_datetime(ticketmaster_df['event_date'], errors='coerce')
            self.df['event_date'] = pd.to_datetime(self.df['event_date'], errors='coerce')
            before = self.df.shape[1]
            self.df = self.df.merge(ticketmaster_df, on=['city', 'event_date'], how='left')
            self._log(f"Merged Ticketmaster data: +{self.df.shape[1] - before} columns")
        else:
            logger.warning("Skipping Ticketmaster merge: missing required columns")

    def merge_eventbrite(self, eventbrite_df: pd.DataFrame) -> None:
        if self.df is None:
            raise ValueError("Primary dataset not loaded")
        if {'city', 'event_date'}.issubset(eventbrite_df.columns):
            eventbrite_df['event_date'] = pd.to_datetime(eventbrite_df['event_date'], errors='coerce')
            self.df['event_date'] = pd.to_datetime(self.df['event_date'], errors='coerce')
            before = self.df.shape[1]
            self.df = self.df.merge(eventbrite_df, on=['city', 'event_date'], how='left')
            self._log(f"Merged Eventbrite data: +{self.df.shape[1] - before} columns")
        else:
            logger.warning("Skipping Eventbrite merge: missing required columns")

    def merge_venue_stats(self, venues_df: pd.DataFrame) -> None:
        if self.df is None:
            raise ValueError("Primary dataset not loaded")
        if 'city' in venues_df.columns:
            before = self.df.shape[1]
            self.df = self.df.merge(venues_df, on='city', how='left')
            self._log(f"Merged venue data: +{self.df.shape[1] - before} columns")
        else:
            logger.warning("Skipping venue merge: missing city column")

    def merge_demographics(self, demographics_df: pd.DataFrame) -> None:
        if self.df is None:
            raise ValueError("Primary dataset not loaded")
        if 'city' in demographics_df.columns:
            before = self.df.shape[1]
            self.df = self.df.merge(demographics_df, on='city', how='left')
            self._log(f"Merged demographics data: +{self.df.shape[1] - before} columns")
        else:
            logger.warning("Skipping demographics merge: missing city column")

    def handle_missing(self, strategy: str = 'median') -> None:
        if self.df is None:
            raise ValueError("Primary dataset not loaded")
        before = len(self.df)
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'ffill':
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        removed = before - len(self.df)
        self._log(f"Missing handling ({strategy}): removed {removed} rows")

    def engineer_post_merge(self) -> None:
        if self.df is None:
            raise ValueError("Primary dataset not loaded")
        if 'median_income' in self.df.columns and 'venue_capacity' in self.df.columns:
            self.df['income_per_capacity'] = self.df['median_income'] / (self.df['venue_capacity'] + 1)
        if 'ticketmaster_event_count' in self.df.columns and 'competing_events' in self.df.columns:
            self.df['ticket_demand_ratio'] = self.df['ticketmaster_event_count'] / (self.df['competing_events'] + 1)
        if 'weather_precipitation' in self.df.columns:
            self.df['rain_weekend'] = ((self.df['weather_precipitation'] > 0) & (self.df['is_weekend'] == 1)).astype(int)

    def get_stats(self) -> Dict:
        if self.df is None:
            return {}
        return {
            'shape': self.df.shape,
            'missing_pct': round((self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100, 2)
        }


def merge_all_datasets(concerts_csv: str, data_dir: str = '.') -> pd.DataFrame:
    logger.info("Starting dataset merge workflow...")
    merger = DatasetMerger()
    merger.load_primary(concerts_csv)

    merges = [
        ('spotify', merger.merge_spotify),
        ('weather', merger.merge_weather),
        ('ticketmaster', merger.merge_ticketmaster),
        ('eventbrite', merger.merge_eventbrite),
        ('venues', merger.merge_venue_stats),
        ('demographics', merger.merge_demographics)
    ]

    for name, method in merges:
        try:
            path = f'{data_dir}/{name}.csv'
            df = pd.read_csv(path)
            method(df)
        except FileNotFoundError:
            logger.warning(f"{name}.csv not found, skipping")
        except Exception as exc:
            logger.error(f"Error merging {name}: {exc}")

    merger.handle_missing(strategy='median')
    merger.engineer_post_merge()
    output_path = f'{data_dir}/concerts_merged.csv'
    merger.df.to_csv(output_path, index=False)
    logger.info(f"Saved merged dataset to {output_path}: {merger.df.shape}")
    return merger.df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    try:
        df = merge_all_datasets('concerts_processed.csv', '.')
        print(f"\n✓ Dataset merge complete: {df.shape}")
    except Exception as exc:
        print(f"Error: {exc}")
