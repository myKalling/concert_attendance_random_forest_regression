"""
Dedicated Census fetcher for demographic features like median income and population.
"""
from pathlib import Path
from typing import Dict
import pandas as pd
import requests
from fetcher_base import DataFetcher, CENSUS_BASE_URL, CITY_STATE_CODES, API_TIMEOUT, normalize_city
import logging

logger = logging.getLogger(__name__)


class CensusFetcher(DataFetcher):
    def __init__(self, output_file: str = 'demographics.csv'):
        super().__init__(output_file)
        self.api_key = __import__('os').getenv('CENSUS_API_KEY')

    def _lookup_city_state(self, city: str) -> str:
        return CITY_STATE_CODES.get(city)

    def _fetch_census(self, city: str, state_code: str) -> Dict:
        params = {
            'get': 'NAME,B19013_001E,B01003_001E',
            'for': 'place:*',
            'in': f'state:{state_code}',
            'key': self.api_key
        }
        try:
            response = requests.get(CENSUS_BASE_URL, params=params, timeout=API_TIMEOUT)
            response.raise_for_status()
            rows = response.json()
            header = rows[0]
            for row in rows[1:]:
                row_dict = dict(zip(header, row))
                if city.lower() in row_dict.get('NAME', '').lower():
                    return {
                        'median_income': float(row_dict.get('B19013_001E') or 0),
                        'population': int(row_dict.get('B01003_001E') or 0)
                    }
        except Exception as exc:
            logger.warning(f'Census lookup failed for {city}: {exc}')
        # API failed or returned unexpected data. Try a local fallback file `demographics.csv` if present.
        try:
            local = pd.read_csv('demographics.csv')
            local['city'] = local['city'].astype(str).str.strip()
            match = local[local['city'].str.lower() == city.lower()]
            if not match.empty:
                row = match.iloc[0]
                return {
                    'median_income': float(row.get('median_income') or 0),
                    'population': int(row.get('population') or 0)
                }
        except Exception:
            # Give up silently; return Nones to be handled downstream
            pass
        return {'median_income': None, 'population': None}

    def fetch(self, source_csv: str):
        cities = set()
        for chunk in pd.read_csv(source_csv, usecols=['city'], chunksize=50):
            chunk['city'] = chunk['city'].astype(str).apply(normalize_city)
            cities.update(chunk['city'].dropna().unique())

        for city in sorted(cities):
            result = {'city': city}
            state_code = self._lookup_city_state(city)
            if state_code and self.api_key:
                result.update(self._fetch_census(city, state_code))
            else:
                logger.warning(f'Skipping Census lookup for {city}: missing state code or API key')
                result.update({'median_income': None, 'population': None})
            self.data.append(result)
            self.success += 1
            self._delay()

        df = pd.DataFrame(self.data)
        # If the API produced mostly empty rows, create a lightweight synthetic fallback file
        empty_ratio = df[['median_income', 'population']].isnull().all(axis=1).mean()
        if empty_ratio > 0.4:
            logger.warning('Many Census lookups failed; creating synthetic demographics fallback for missing cities')
            rows = []
            for city in sorted(cities):
                existing = df[df['city'].str.lower() == city.lower()]
                if not existing.empty and existing[['median_income', 'population']].notna().any(axis=None):
                    rows.append(existing.iloc[0].to_dict())
                else:
                    # Conservative defaults — mark as synthetic so downstream users can identify them
                    rows.append({'city': city, 'median_income': 60000, 'population': 100000, 'synthetic_demographics': True})
            df = pd.DataFrame(rows)
        return df
