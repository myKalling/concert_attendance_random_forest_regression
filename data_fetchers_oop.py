"""
Primary orchestrator for external data fetchers.
"""
import logging
from typing import Dict

import pandas as pd

from fetcher_base import normalize_city
from eventbrite_fetcher import EventbriteFetcher
from census_fetcher import CensusFetcher
from fetcher_base import DataFetcher
from fetcher_base import API_TIMEOUT, SPOTIFY_TOKEN_URL, SPOTIFY_SEARCH_URL
from fetcher_base import OPEN_METEO_GEOCODE, OPEN_METEO_ARCHIVE, OPEN_METEO_FORECAST
from fetcher_base import TICKETMASTER_EVENTS_URL, TICKETMASTER_VENUE_URL, MUSICBRAINZ_ARTIST_SEARCH
from fetcher_base import OUTDOOR_KEYWORDS

import requests
import os
import base64
import time
from datetime import datetime


class SetlistFetcher(DataFetcher):
    def __init__(self, output_file: str = 'setlistfm.csv'):
        super().__init__(output_file)
        self.api_key = os.getenv('SETLISTFM_API_KEY')

    def _query_artist(self, artist: str) -> Dict:
        if not self.api_key:
            logger.warning('Setlist.fm key missing; skipping setlist enrichment')
            return {'artist': artist, 'setlist_recent_count': None}
        url = 'https://api.setlist.fm/rest/1.0/search/artists'
        headers = {'x-api-key': self.api_key, 'Accept': 'application/json', 'User-Agent': 'concert-pipeline/1.0'}
        params = {'artistName': artist, 'p': 1}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            total = data.get('total', None)
            return {'artist': artist, 'setlist_recent_count': total}
        except Exception as exc:
            logger.warning(f'Setlist.fm error for {artist}: {exc}')
            return {'artist': artist, 'setlist_recent_count': None}

    def _resolve_artists(self, source_csv: str):
        artists = []
        try:
            df = pd.read_csv(source_csv, usecols=['artist'])
            artists = df['artist'].dropna().astype(str).unique().tolist()
        except Exception:
            pass

        if not artists:
            if os.path.exists('spotify.csv'):
                try:
                    spotify = pd.read_csv('spotify.csv')
                    if 'spotify_top_artist' in spotify.columns:
                        artists = spotify['spotify_top_artist'].dropna().astype(str).unique().tolist()
                except Exception:
                    pass

        return sorted(set(artists))

    def fetch(self, source_csv: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        artists = self._resolve_artists(source_csv)
        for artist in artists:
            row = self._query_artist(artist)
            self.data.append(row)
            self.success += 1
            self._delay()
        return pd.DataFrame(self.data)


class TicketAvailabilityFetcher(DataFetcher):
    def __init__(self, output_file: str = 'ticket_availability.csv'):
        super().__init__(output_file)
        self.api_key = os.getenv('TICKETMASTER_API_KEY')

    def _check_availability(self, city: str, event_date: str, artist: str = None) -> Dict:
        if not self.api_key:
            return {'city': city, 'event_date': event_date, 'artist': artist, 'has_tickets': None, 'ticket_est_count': None}
        params = {'apikey': self.api_key, 'city': city, 'startDateTime': f'{event_date}T00:00:00Z', 'endDateTime': f'{event_date}T23:59:59Z', 'size': 20}
        if artist:
            params['keyword'] = artist
        try:
            resp = requests.get(TICKETMASTER_EVENTS_URL, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            events = resp.json().get('_embedded', {}).get('events', [])
            if not events:
                return {'city': city, 'event_date': event_date, 'artist': artist, 'has_tickets': 0, 'ticket_est_count': 0}
            ticket_counts = 0
            for ev in events:
                # estimate tickets by number of priceRanges entries
                ticket_counts += len(ev.get('priceRanges') or [])
            return {'city': city, 'event_date': event_date, 'artist': artist, 'has_tickets': 1, 'ticket_est_count': ticket_counts}
        except Exception as exc:
            logger.warning(f'Ticket availability error for {city} {event_date} {artist}: {exc}')
            return {'city': city, 'event_date': event_date, 'artist': artist, 'has_tickets': None, 'ticket_est_count': None}

    def fetch(self, source_csv: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        df = pd.read_csv(source_csv, usecols=['city', 'event_date', 'artist'] if 'artist' in pd.read_csv(source_csv, nrows=0).columns else ['city', 'event_date'])
        df['city'] = df['city'].astype(str).apply(normalize_city)
        for row in df.drop_duplicates().itertuples(index=False):
            artist = getattr(row, 'artist', None) if 'artist' in df.columns else None
            res = self._check_availability(row.city, str(row.event_date), artist)
            self.data.append(res)
            self.success += 1
            self._delay()
        return pd.DataFrame(self.data)

logger = logging.getLogger(__name__)


class SpotifyGenreFetcher(DataFetcher):
    def __init__(self, output_file: str = 'spotify.csv'):
        super().__init__(output_file)
        self.token = None
        self.client_id = os.getenv('SPOTIPY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')

    def _get_token(self) -> str | None:
        if self.token:
            return self.token
        if not self.client_id or not self.client_secret:
            logger.warning('Spotify credentials missing')
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
            logger.info('Spotify token acquired')
            return self.token
        except Exception as exc:
            logger.error(f'Spotify token error: {exc}')
            return None

    def _search_top_artist(self, genre: str) -> Dict:
        token = self._get_token()
        if not token:
            return {'genre': genre, 'spotify_top_artist': None, 'spotify_genre_popularity': None, 'spotify_genre_followers': None}
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

    def fetch(self, source_csv: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        df = pd.read_csv(source_csv, usecols=['genre'])
        genres = sorted(df['genre'].dropna().astype(str).unique())
        for genre in genres:
            row = self._search_top_artist(genre)
            self.data.append(row)
            self.success += 1
            self._delay()
        return pd.DataFrame(self.data)


class SpotifyArtistFetcher(DataFetcher):
    def __init__(self, output_file: str = 'spotify_artists.csv'):
        super().__init__(output_file)
        self.token = None
        self.client_id = os.getenv('SPOTIPY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')

    def _get_token(self) -> str | None:
        if self.token:
            return self.token
        if not self.client_id or not self.client_secret:
            logger.warning('Spotify credentials missing; skipping genre artist enrichment')
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
            logger.info('Spotify token acquired for genre artist lookup')
            return self.token
        except Exception as exc:
            logger.error(f'Spotify token error: {exc}')
            return None

    def _search_artist_for_genre(self, genre: str) -> Dict:
        token = self._get_token()
        if not token:
            return {'genre': genre, 'spotify_genre_artist': None, 'spotify_genre_artist_popularity': None, 'spotify_genre_artist_followers': None, 'spotify_genre_artist_genres': None}
        try:
            resp = requests.get(
                SPOTIFY_SEARCH_URL,
                headers={'Authorization': f'Bearer {token}'},
                params={'q': genre, 'type': 'artist', 'limit': 1},
                timeout=API_TIMEOUT
            )
            resp.raise_for_status()
            items = resp.json().get('artists', {}).get('items', [])
            if not items:
                return {'genre': genre, 'spotify_genre_artist': None, 'spotify_genre_artist_popularity': None, 'spotify_genre_artist_followers': None, 'spotify_genre_artist_genres': None}
            artist_info = items[0]
            return {
                'genre': genre,
                'spotify_genre_artist': artist_info.get('name'),
                'spotify_genre_artist_popularity': artist_info.get('popularity'),
                'spotify_genre_artist_followers': artist_info.get('followers', {}).get('total'),
                'spotify_genre_artist_genres': '|'.join(artist_info.get('genres', [])) if artist_info.get('genres') else None
            }
        except Exception as exc:
            logger.warning(f"Spotify genre artist search error for '{genre}': {exc}")
            return {'genre': genre, 'spotify_genre_artist': None, 'spotify_genre_artist_popularity': None, 'spotify_genre_artist_followers': None, 'spotify_genre_artist_genres': None}

    def fetch(self, source_csv: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(source_csv, usecols=['genre'])
            genres = sorted(df['genre'].dropna().astype(str).unique())
        except Exception:
            genres = []

        for genre in genres:
            row = self._search_artist_for_genre(genre)
            self.data.append(row)
            self.success += 1
            self._delay()
        return pd.DataFrame(self.data)


class MusicBrainzFetcher(DataFetcher):
    def __init__(self, output_file: str = 'musicbrainz.csv'):
        super().__init__(output_file)

    def _search_genre(self, genre: str) -> Dict:
        headers = {'User-Agent': 'concert-pipeline/1.0'}
        params = {'query': genre, 'fmt': 'json', 'limit': 1}
        try:
            resp = requests.get(MUSICBRAINZ_ARTIST_SEARCH, headers=headers, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            artists = resp.json().get('artists', [])
            if not artists:
                return {'genre': genre, 'mbid': None, 'mb_area': None, 'mb_begin': None, 'mb_end': None, 'mb_tags': None}
            top = artists[0]
            tags = top.get('tags', [])
            tag_names = '|'.join([t.get('name') for t in tags if t.get('name')]) if tags else None
            return {
                'genre': genre,
                'mbid': top.get('id'),
                'mb_area': top.get('area', {}).get('name'),
                'mb_begin': top.get('life-span', {}).get('begin'),
                'mb_end': top.get('life-span', {}).get('end'),
                'mb_tags': tag_names
            }
        except Exception as exc:
            logger.warning(f'MusicBrainz error for genre {genre}: {exc}')
            return {'genre': genre, 'mbid': None, 'mb_area': None, 'mb_begin': None, 'mb_end': None, 'mb_tags': None}

    def fetch(self, source_csv: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(source_csv, usecols=['genre'])
            genres = sorted(df['genre'].dropna().astype(str).unique())
        except Exception:
            genres = []

        for genre in genres:
            row = self._search_genre(genre)
            self.data.append(row)
            self.success += 1
            self._delay()
        return pd.DataFrame(self.data)


class WeatherFetcher(DataFetcher):
    def __init__(self, output_file: str = 'weather.csv'):
        super().__init__(output_file)
        self.geocode_cache = {}

    def _geocode_city(self, city: str):
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
            coords = {
                'latitude': results[0]['latitude'],
                'longitude': results[0]['longitude']
            }
            self.geocode_cache[city] = coords
            return coords
        except Exception as exc:
            logger.warning(f'Geocode error for {city}: {exc}')
            return None

    def _fetch_weather(self, latitude: float, longitude: float, event_date: str) -> Dict:
        target_date = datetime.strptime(event_date, '%Y-%m-%d').date()
        url = OPEN_METEO_ARCHIVE if target_date <= datetime.utcnow().date() else OPEN_METEO_FORECAST
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'daily': 'temperature_2m_max,precipitation_sum,windspeed_10m_max',
            'start_date': event_date,
            'end_date': event_date,
            'timezone': 'UTC'
        }
        try:
            resp = requests.get(url, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            daily = resp.json().get('daily', {})
            return {
                'weather_temp_max': daily.get('temperature_2m_max', [None])[0],
                'weather_precipitation': daily.get('precipitation_sum', [None])[0],
                'weather_windspeed_max': daily.get('windspeed_10m_max', [None])[0]
            }
        except Exception as exc:
            logger.warning(f'Weather fetch failed for {event_date} {latitude},{longitude}: {exc}')
            return {'weather_temp_max': None, 'weather_precipitation': None, 'weather_windspeed_max': None}

    def fetch(self, source_csv: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        df = pd.read_csv(source_csv, usecols=['city', 'event_date'])
        df['city'] = df['city'].astype(str).apply(normalize_city)
        for row in df.drop_duplicates().itertuples(index=False):
            coords = self._geocode_city(row.city)
            result = {'city': row.city, 'event_date': str(row.event_date)}
            if coords:
                result.update(self._fetch_weather(coords['latitude'], coords['longitude'], result['event_date']))
            else:
                result.update({'weather_temp_max': None, 'weather_precipitation': None, 'weather_windspeed_max': None})
            self.data.append(result)
            self.success += 1
            self._delay()
        return pd.DataFrame(self.data)


class GeoFetcher(DataFetcher):
    def __init__(self, output_file: str = 'geodata.csv'):
        super().__init__(output_file)
        self.geocode_cache = {}

    def _geocode_city(self, city: str):
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
            data = results[0]
            payload = {
                'city': city,
                'latitude': data.get('latitude'),
                'longitude': data.get('longitude'),
                'country': data.get('country'),
                'country_code': data.get('country_code'),
                'timezone': data.get('timezone'),
                'elevation': data.get('elevation')
            }
            self.geocode_cache[city] = payload
            return payload
        except Exception as exc:
            logger.warning(f'GeoFetcher error for {city}: {exc}')
            return None

    def fetch(self, source_csv: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        df = pd.read_csv(source_csv, usecols=['city'])
        cities = sorted(df['city'].astype(str).apply(normalize_city).dropna().unique())
        for city in cities:
            result = self._geocode_city(city)
            if result:
                self.data.append(result)
                self.success += 1
            else:
                self.data.append({'city': city, 'latitude': None, 'longitude': None, 'country': None, 'country_code': None, 'timezone': None, 'elevation': None})
            self._delay()
        return pd.DataFrame(self.data)


class TicketmasterFetcher(DataFetcher):
    def __init__(self, output_file: str = 'ticketmaster.csv'):
        super().__init__(output_file)
        self.api_key = os.getenv('TICKETMASTER_API_KEY')

    def _query_events(self, city: str, event_date: str) -> Dict:
        if not self.api_key:
            logger.warning('Ticketmaster key missing; skipping Ticketmaster enrichment')
            return {'ticketmaster_event_count': 0, 'ticketmaster_avg_min_price': None, 'ticketmaster_avg_max_price': None, 'ticketmaster_venue_count': None}
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
            prices = []
            venues = set()
            for event in events:
                for venue in event.get('_embedded', {}).get('venues', []):
                    if venue.get('name'):
                        venues.add(venue.get('name'))
                for pr in event.get('priceRanges', []) or []:
                    prices.append((pr.get('min'), pr.get('max')))
            valid_min = [p[0] for p in prices if p[0] is not None]
            valid_max = [p[1] for p in prices if p[1] is not None]
            return {
                'ticketmaster_event_count': len(events),
                'ticketmaster_avg_min_price': (sum(valid_min) / len(valid_min)) if valid_min else None,
                'ticketmaster_avg_max_price': (sum(valid_max) / len(valid_max)) if valid_max else None,
                'ticketmaster_venue_count': len(venues)
            }
        except Exception as exc:
            logger.warning(f'Ticketmaster error for {city} on {event_date}: {exc}')
            return {'ticketmaster_event_count': 0, 'ticketmaster_avg_min_price': None, 'ticketmaster_avg_max_price': None, 'ticketmaster_venue_count': None}

    def fetch(self, source_csv: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        df = pd.read_csv(source_csv, usecols=['city', 'event_date'])
        df['city'] = df['city'].astype(str).apply(normalize_city)
        for row in df.drop_duplicates().itertuples(index=False):
            result = {'city': row.city, 'event_date': str(row.event_date)}
            result.update(self._query_events(row.city, result['event_date']))
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
        return int(any(keyword in name.lower() for keyword in OUTDOOR_KEYWORDS))

    def _query_venues(self, city: str) -> Dict:
        if not self.api_key:
            logger.warning('Ticketmaster key missing; skipping venue enrichment')
            return {'venue_count': 0, 'venue_outdoor_pct': 0.0}
        params = {'apikey': self.api_key, 'city': city, 'size': 50}
        try:
            resp = requests.get(TICKETMASTER_VENUE_URL, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            venues = resp.json().get('_embedded', {}).get('venues', [])
            if not venues:
                return {'venue_count': 0, 'venue_outdoor_pct': 0.0}
            flags = [self._infer_outdoor(v.get('name')) for v in venues]
            return {'venue_count': len(venues), 'venue_outdoor_pct': sum(flags) / len(flags)}
        except Exception as exc:
            logger.warning(f'Ticketmaster venue error for {city}: {exc}')
            return {'venue_count': 0, 'venue_outdoor_pct': 0.0}

    def fetch(self, source_csv: str) -> pd.DataFrame:
        df = pd.read_csv(source_csv, usecols=['city'])
        cities = sorted(df['city'].astype(str).apply(normalize_city).dropna().unique())
        for city in cities:
            result = {'city': city}
            result.update(self._query_venues(city))
            self.data.append(result)
            self.success += 1
            self._delay()
        return pd.DataFrame(self.data)


class BandsintownFetcher(DataFetcher):
    """Fetch simple artist event counts from Bandsintown.
    Uses mapped Spotify top artist per-genre when primary data lacks artist.
    """
    BASE_URL = 'https://rest.bandsintown.com/artists/{artist}/events'

    def __init__(self, output_file: str = 'bandsintown.csv'):
        super().__init__(output_file)
        self.app_id = os.getenv('BANDSINTOWN_APP_ID', 'concert-pipeline')

    def _query_artist(self, artist: str) -> Dict:
        if not artist:
            return {'bandsintown_artist': None, 'bandsintown_event_count': None, 'bandsintown_upcoming_count': None}
        try:
            url = self.BASE_URL.format(artist=artist)
            resp = requests.get(url, params={'app_id': self.app_id}, timeout=API_TIMEOUT)
            resp.raise_for_status()
            events = resp.json()
            if isinstance(events, dict) and events.get('errors'):
                return {'bandsintown_artist': artist, 'bandsintown_event_count': 0, 'bandsintown_upcoming_count': 0}
            total = len(events) if isinstance(events, list) else 0
            # Bandsintown returns upcoming events by default; treat length as upcoming
            return {'bandsintown_artist': artist, 'bandsintown_event_count': total, 'bandsintown_upcoming_count': total}
        except Exception as exc:
            logger.warning(f'Bandsintown error for {artist}: {exc}')
            return {'bandsintown_artist': artist, 'bandsintown_event_count': None, 'bandsintown_upcoming_count': None}

    def fetch(self, source_csv: str) -> pd.DataFrame:
        # Try to resolve genre -> top artist mapping from spotify.csv
        genres = []
        try:
            df = pd.read_csv(source_csv, usecols=['genre'])
            genres = sorted(df['genre'].dropna().astype(str).unique())
        except Exception:
            genres = []

        mapping = {}
        if os.path.exists('spotify.csv'):
            try:
                s = pd.read_csv('spotify.csv')
                if 'genre' in s.columns and 'spotify_top_artist' in s.columns:
                    mapping = dict(s.dropna(subset=['genre', 'spotify_top_artist']).drop_duplicates().set_index('genre')['spotify_top_artist'])
            except Exception:
                mapping = {}

        for genre in genres:
            artist = mapping.get(genre)
            row = {'genre': genre}
            if artist:
                row.update(self._query_artist(artist))
            else:
                row.update({'bandsintown_artist': None, 'bandsintown_event_count': None, 'bandsintown_upcoming_count': None})
            self.data.append(row)
            self.success += 1
            self._delay()
        return pd.DataFrame(self.data)


def fetch_all_external_data(concerts_csv: str, output_dir: str = '.') -> Dict[str, pd.DataFrame]:
    logger.info('Fetching all external data sources...')
    fetchers = [
        SpotifyGenreFetcher(f'{output_dir}/spotify.csv'),
        SpotifyArtistFetcher(f'{output_dir}/spotify_artists.csv'),
        MusicBrainzFetcher(f'{output_dir}/musicbrainz.csv'),
        WeatherFetcher(f'{output_dir}/weather.csv'),
        GeoFetcher(f'{output_dir}/geodata.csv'),
        TicketmasterFetcher(f'{output_dir}/ticketmaster.csv'),
        EventbriteFetcher(f'{output_dir}/eventbrite.csv'),
        VenueFetcher(f'{output_dir}/venues.csv'),
        CensusFetcher(f'{output_dir}/demographics.csv'),
        SetlistFetcher(f'{output_dir}/setlistfm.csv'),
        TicketAvailabilityFetcher(f'{output_dir}/ticket_availability.csv')
    ]
    # Bandsintown: per-genre artist event counts via mapped Spotify top artist
    fetchers.insert(2, BandsintownFetcher(f'{output_dir}/bandsintown.csv'))
    results = {}
    for fetcher in fetchers:
        try:
            df = fetcher.fetch(concerts_csv)
            fetcher.save()
            results[fetcher.output_file.stem] = df
        except Exception as exc:
            logger.error(f'Error fetching {fetcher.__class__.__name__}: {exc}')
    logger.info(f'✓ Fetched %d sources', len(results))
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    fetch_all_external_data('concerts_processed.csv', '.')
