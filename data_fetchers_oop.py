"""
Data Fetchers: External Data Acquisition using OOP with Abstract Base Class
"""

import os
import base64
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime, date

import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)

API_TIMEOUT = 12
RATE_LIMIT_DELAY = 0.5
SPOTIFY_TOKEN_URL = 'https://accounts.spotify.com/api/token'
SPOTIFY_SEARCH_URL = 'https://api.spotify.com/v1/search'
OPEN_METEO_GEOCODE = 'https://geocoding-api.open-meteo.com/v1/search'
OPEN_METEO_ARCHIVE = 'https://archive-api.open-meteo.com/v1/archive'
OPEN_METEO_FORECAST = 'https://api.open-meteo.com/v1/forecast'
TICKETMASTER_EVENTS_URL = 'https://app.ticketmaster.com/discovery/v2/events.json'
TICKETMASTER_VENUE_URL = 'https://app.ticketmaster.com/discovery/v2/venues.json'
EVENTBRITE_SEARCH_URL = 'https://www.eventbriteapi.com/v3/events/search/'
CENSUS_BASE_URL = 'https://api.census.gov/data/2022/acs/acs5'

CITY_STATE_CODES = {
    'New York': '36',
    'Los Angeles': '06',
    'Chicago': '17',
    'San Francisco': '06',
    'Denver': '08'
}

OUTDOOR_KEYWORDS = ['park', 'stadium', 'amphitheater', 'ballpark', 'woodland', 'plaza']


class DataFetcher(ABC):
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.data = []
        self.errors = 0
        self.success = 0

    @abstractmethod
    def fetch(self, source_csv: str) -> pd.DataFrame:
        pass

    def save(self) -> None:
        if not self.data:
            logger.warning(f"No data to save for {self.output_file}")
            return
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_file, index=False)
        logger.info(f"Saved {len(df)} rows to {self.output_file}")

    def _delay(self):
        time.sleep(RATE_LIMIT_DELAY)


class SpotifyGenreFetcher(DataFetcher):
    def __init__(self, output_file: str = 'spotify.csv'):
        super().__init__(output_file)
        self.token = None
        self.client_id = os.getenv('SPOTIPY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')

    def _get_token(self) -> Optional[str]:
        if self.token:
            return self.token
        if not self.client_id or not self.client_secret:
            logger.warning("Spotify credentials missing")
            return None

        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        try:
            resp = requests.post(
                SPOTIFY_TOKEN_URL,
                headers={'Authorization': f'Basic {auth_header}'},
                data={'grant_type': 'client_credentials'},
                timeout=API_TIMEOUT
            )
            resp.raise_for_status()
            self.token = resp.json().get('access_token')
            logger.info("Spotify token acquired")
            return self.token
        except Exception as exc:
            logger.error(f"Spotify token error: {exc}")
            return None

    def _search_top_artist(self, genre: str) -> Dict:
        token = self._get_token()
        if not token:
            return {
                'genre': genre,
                'spotify_top_artist': None,
                'spotify_genre_popularity': None,
                'spotify_genre_followers': None
            }

        params = {'q': genre, 'type': 'artist', 'limit': 1}
        try:
            resp = requests.get(
                SPOTIFY_SEARCH_URL,
                headers={'Authorization': f'Bearer {token}'},
                params=params,
                timeout=API_TIMEOUT
            )
            resp.raise_for_status()
            items = resp.json().get('artists', {}).get('items', [])
            if not items:
                return {'genre': genre, 'spotify_top_artist': None, 'spotify_genre_popularity': None, 'spotify_genre_followers': None}
            artist = items[0]
            return {
                'genre': genre,
                'spotify_top_artist': artist.get('name'),
                'spotify_genre_popularity': artist.get('popularity'),
                'spotify_genre_followers': artist.get('followers', {}).get('total')
            }
        except Exception as exc:
            logger.warning(f"Spotify search error for genre '{genre}': {exc}")
            return {'genre': genre, 'spotify_top_artist': None, 'spotify_genre_popularity': None, 'spotify_genre_followers': None}

    def fetch(self, source_csv: str) -> pd.DataFrame:
        df = pd.read_csv(source_csv)
        if 'genre' not in df.columns:
            logger.warning("No genre column for Spotify fetch")
            return pd.DataFrame()

        genres = sorted(df['genre'].dropna().unique())
        for genre in genres:
            row = self._search_top_artist(genre)
            self.data.append(row)
            self.success += 1
            self._delay()

        if not self.data:
            logger.warning("Spotify genre fetch returned no rows")
        return pd.DataFrame(self.data)


class WeatherFetcher(DataFetcher):
    def __init__(self, output_file: str = 'weather.csv'):
        super().__init__(output_file)
        self.geocode_cache = {}

    def _geocode_city(self, city: str) -> Optional[Dict]:
        if city in self.geocode_cache:
            return self.geocode_cache[city]

        try:
            resp = requests.get(
                OPEN_METEO_GEOCODE,
                params={'name': city, 'count': 1, 'language': 'en'},
                timeout=API_TIMEOUT
            )
            resp.raise_for_status()
            results = resp.json().get('results')
            if not results:
                return None
            entry = results[0]
            coords = {
                'latitude': entry['latitude'],
                'longitude': entry['longitude'],
                'name': entry['name']
            }
            self.geocode_cache[city] = coords
            return coords
        except Exception as exc:
            logger.error(f"Geocode error for {city}: {exc}")
            return None

    def _fetch_weather(self, latitude: float, longitude: float, event_date: str) -> Dict:
        target_date = datetime.strptime(event_date, '%Y-%m-%d').date()
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'daily': 'temperature_2m_max,precipitation_sum,windspeed_10m_max',
            'timezone': 'UTC'
        }
        if target_date <= date.today():
            params.update({'start_date': event_date, 'end_date': event_date})
            url = OPEN_METEO_ARCHIVE
        else:
            params.update({'start_date': event_date, 'end_date': event_date})
            url = OPEN_METEO_FORECAST

        try:
            resp = requests.get(url, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            daily = resp.json().get('daily', {})
            if not daily or not daily.get('time'):
                return {}
            return {
                'weather_temp_max': daily.get('temperature_2m_max', [None])[0],
                'weather_precipitation': daily.get('precipitation_sum', [None])[0],
                'weather_windspeed_max': daily.get('windspeed_10m_max', [None])[0]
            }
        except Exception as exc:
            logger.warning(f"Weather fetch failed for {event_date} {latitude},{longitude}: {exc}")
            return {}

    def fetch(self, source_csv: str) -> pd.DataFrame:
        df = pd.read_csv(source_csv)
        if not {'city', 'event_date'}.issubset(df.columns):
            logger.warning("City/date required for weather fetch")
            return pd.DataFrame()

        combos = df[['city', 'event_date']].drop_duplicates().reset_index(drop=True)
        for row in combos.itertuples(index=False):
            result = {'city': row.city, 'event_date': row.event_date}
            coords = self._geocode_city(row.city)
            if coords:
                weather = self._fetch_weather(coords['latitude'], coords['longitude'], row.event_date)
                result.update(weather)
            else:
                result.update({'weather_temp_max': None, 'weather_precipitation': None, 'weather_windspeed_max': None})
            self.data.append(result)
            self.success += 1
            self._delay()

        return pd.DataFrame(self.data)


class TicketmasterFetcher(DataFetcher):
    def __init__(self, output_file: str = 'ticketmaster.csv'):
        super().__init__(output_file)
        self.api_key = os.getenv('TICKETMASTER_API_KEY')

    def _query_events(self, city: str, event_date: str) -> Dict:
        params = {
            'apikey': self.api_key,
            'city': city,
            'sort': 'date,asc',
            'startDateTime': f'{event_date}T00:00:00Z',
            'endDateTime': f'{event_date}T23:59:59Z',
            'size': 20
        }
        try:
            resp = requests.get(TICKETMASTER_EVENTS_URL, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            events = payload.get('_embedded', {}).get('events', [])
            if not events:
                return {'ticketmaster_event_count': 0, 'ticketmaster_avg_min_price': None, 'ticketmaster_avg_max_price': None}

            prices = []
            venues = set()
            for event in events:
                for venue in event.get('_embedded', {}).get('venues', []):
                    venues.add(venue.get('name'))
                if event.get('priceRanges'):
                    for pr in event.get('priceRanges', []):
                        prices.append((pr.get('min'), pr.get('max')))

            avg_min = None
            avg_max = None
            if prices:
                valid_min = [p[0] for p in prices if p[0] is not None]
                valid_max = [p[1] for p in prices if p[1] is not None]
                avg_min = sum(valid_min) / len(valid_min) if valid_min else None
                avg_max = sum(valid_max) / len(valid_max) if valid_max else None

            return {
                'ticketmaster_event_count': len(events),
                'ticketmaster_avg_min_price': avg_min,
                'ticketmaster_avg_max_price': avg_max,
                'ticketmaster_venue_count': len(venues)
            }
        except Exception as exc:
            logger.warning(f"Ticketmaster error for {city} on {event_date}: {exc}")
            return {'ticketmaster_event_count': 0, 'ticketmaster_avg_min_price': None, 'ticketmaster_avg_max_price': None, 'ticketmaster_venue_count': None}

    def fetch(self, source_csv: str) -> pd.DataFrame:
        df = pd.read_csv(source_csv)
        if 'city' not in df.columns or 'event_date' not in df.columns:
            return pd.DataFrame()

        combos = df[['city', 'event_date']].drop_duplicates().reset_index(drop=True)
        for row in combos.itertuples(index=False):
            result = {'city': row.city, 'event_date': row.event_date}
            if self.api_key:
                result.update(self._query_events(row.city, row.event_date))
            else:
                result.update({'ticketmaster_event_count': 0, 'ticketmaster_avg_min_price': None, 'ticketmaster_avg_max_price': None, 'ticketmaster_venue_count': None})
            self.data.append(result)
            self.success += 1
            self._delay()

        return pd.DataFrame(self.data)


class EventbriteFetcher(DataFetcher):
    def __init__(self, output_file: str = 'eventbrite.csv'):
        super().__init__(output_file)
        self.token = os.getenv('EVENTBRITE_TOKEN')

    def _query_events(self, city: str, event_date: str) -> Dict:
        params = {
            'location.address': city,
            'start_date.range_start': f'{event_date}T00:00:00Z',
            'start_date.range_end': f'{event_date}T23:59:59Z',
            'expand': 'venue',
            'page': 1
        }
        headers = {'Authorization': f'Bearer {self.token}'} if self.token else {}
        try:
            resp = requests.get(EVENTBRITE_SEARCH_URL, params=params, headers=headers, timeout=API_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            events = payload.get('events', [])
            free_count = sum(1 for event in events if event.get('is_free'))
            total = len(events)
            return {
                'eventbrite_event_count': total,
                'eventbrite_free_event_pct': (free_count / total) if total else 0.0,
                'eventbrite_nonfree_count': total - free_count
            }
        except Exception as exc:
            logger.warning(f"Eventbrite error for {city} on {event_date}: {exc}")
            return {'eventbrite_event_count': 0, 'eventbrite_free_event_pct': 0.0, 'eventbrite_nonfree_count': 0}

    def fetch(self, source_csv: str) -> pd.DataFrame:
        df = pd.read_csv(source_csv)
        if 'city' not in df.columns or 'event_date' not in df.columns:
            return pd.DataFrame()

        combos = df[['city', 'event_date']].drop_duplicates().reset_index(drop=True)
        for row in combos.itertuples(index=False):
            result = {'city': row.city, 'event_date': row.event_date}
            result.update(self._query_events(row.city, row.event_date))
            self.data.append(result)
            self.success += 1
            self._delay()

        return pd.DataFrame(self.data)


class VenueFetcher(DataFetcher):
    def __init__(self, output_file: str = 'venues.csv'):
        super().__init__(output_file)
        self.api_key = os.getenv('TICKETMASTER_API_KEY')

    def _infer_outdoor(self, name: str) -> int:
        if not name:
            return 0
        lowered = name.lower()
        return int(any(keyword in lowered for keyword in OUTDOOR_KEYWORDS))

    def _query_venues(self, city: str) -> Dict:
        params = {'apikey': self.api_key, 'city': city, 'size': 50}
        try:
            resp = requests.get(TICKETMASTER_VENUE_URL, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            venues = resp.json().get('_embedded', {}).get('venues', [])
            if not venues:
                return {'venue_count': 0, 'venue_outdoor_pct': 0.0}

            outdoor_flags = [self._infer_outdoor(v.get('name')) for v in venues]
            return {
                'venue_count': len(venues),
                'venue_outdoor_pct': sum(outdoor_flags) / len(outdoor_flags)
            }
        except Exception as exc:
            logger.warning(f"Ticketmaster venue error for {city}: {exc}")
            return {'venue_count': 0, 'venue_outdoor_pct': 0.0}

    def fetch(self, source_csv: str) -> pd.DataFrame:
        df = pd.read_csv(source_csv)
        if 'city' not in df.columns:
            return pd.DataFrame()

        for city in sorted(df['city'].dropna().unique()):
            result = {'city': city}
            if self.api_key:
                result.update(self._query_venues(city))
            else:
                result.update({'venue_count': 0, 'venue_outdoor_pct': 0.0})
            self.data.append(result)
            self.success += 1
            self._delay()

        return pd.DataFrame(self.data)


class DemographicsFetcher(DataFetcher):
    def __init__(self, output_file: str = 'demographics.csv'):
        super().__init__(output_file)
        self.api_key = os.getenv('CENSUS_API_KEY')

    def _lookup_city_state(self, city: str) -> Optional[str]:
        return CITY_STATE_CODES.get(city)

    def _fetch_census(self, city: str, state_code: str) -> Dict:
        params = {
            'get': 'NAME,B19013_001E,B01003_001E',
            'for': 'place:*',
            'in': f'state:{state_code}',
            'key': self.api_key
        }
        try:
            resp = requests.get(CENSUS_BASE_URL, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            rows = resp.json()
            header = rows[0]
            results = rows[1:]
            for row in results:
                row_dict = dict(zip(header, row))
                if city.lower() in row_dict['NAME'].lower():
                    return {
                        'median_income': float(row_dict.get('B19013_001E') or 0),
                        'population': int(row_dict.get('B01003_001E') or 0)
                    }
        except Exception as exc:
            logger.warning(f"Census lookup failed for {city}: {exc}")
        return {'median_income': None, 'population': None}

    def fetch(self, source_csv: str) -> pd.DataFrame:
        df = pd.read_csv(source_csv)
        if 'city' not in df.columns:
            return pd.DataFrame()

        for city in sorted(df['city'].dropna().unique()):
            result = {'city': city}
            state_code = self._lookup_city_state(city)
            if state_code and self.api_key:
                result.update(self._fetch_census(city, state_code))
            else:
                result.update({'median_income': None, 'population': None})
            self.data.append(result)
            self.success += 1
            self._delay()

        return pd.DataFrame(self.data)


def fetch_all_external_data(concerts_csv: str, output_dir: str = '.') -> Dict[str, pd.DataFrame]:
    logger.info("Fetching all external data sources...")
    fetchers = [
        SpotifyGenreFetcher(f'{output_dir}/spotify.csv'),
        WeatherFetcher(f'{output_dir}/weather.csv'),
        TicketmasterFetcher(f'{output_dir}/ticketmaster.csv'),
        EventbriteFetcher(f'{output_dir}/eventbrite.csv'),
        VenueFetcher(f'{output_dir}/venues.csv'),
        DemographicsFetcher(f'{output_dir}/demographics.csv'),
    ]

    results = {}
    for fetcher in fetchers:
        try:
            df = fetcher.fetch(concerts_csv)
            fetcher.save()
            results[fetcher.output_file.stem] = df
        except Exception as exc:
            logger.error(f"Error fetching {fetcher.__class__.__name__}: {exc}")
    logger.info(f"✓ Fetched {len(results)} sources")
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    fetch_all_external_data('concerts_raw.csv', '.')
