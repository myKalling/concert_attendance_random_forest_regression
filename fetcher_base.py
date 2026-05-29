"""
Shared base classes and constants for all external data fetchers.
"""
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
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
MUSICBRAINZ_ARTIST_SEARCH = 'https://musicbrainz.org/ws/2/artist/'
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

CITY_NORMALIZATION = {
    'la': 'Los Angeles',
    'l.a.': 'Los Angeles',
    'la, ca': 'Los Angeles',
    'nyc': 'New York',
    'ny': 'New York',
    'new york city': 'New York',
    'new york, ny': 'New York',
    'chgo': 'Chicago',
    'chi': 'Chicago',
    'chicago, il': 'Chicago',
    'sf': 'San Francisco',
    'san francisco, ca': 'San Francisco',
}


def normalize_city(city: str) -> str:
    if not isinstance(city, str):
        return ''
    key = city.strip().lower()
    return CITY_NORMALIZATION.get(key, city.strip().title())


class DataFetcher(ABC):
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.data = []
        self.errors = 0
        self.success = 0

    @abstractmethod
    def fetch(self, source_csv: str, start_date: str = None, end_date: str = None):
        """Fetch external data.
        Optional `start_date` and `end_date` (YYYY-MM-DD) instruct fetchers to
        expand results across a date range when supported.
        """
        pass

    def save(self):
        if not self.data:
            logger.warning(f"No data to save for {self.output_file}")
            return
        df = self.to_dataframe()
        df.to_csv(self.output_file, index=False)
        logger.info(f"Saved {len(df)} rows to {self.output_file}")

    def to_dataframe(self):
        return __import__('pandas').DataFrame(self.data)

    def _delay(self):
        time.sleep(RATE_LIMIT_DELAY)
