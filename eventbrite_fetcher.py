"""
Dedicated Eventbrite fetcher for concert date and city event listings.
"""
from typing import Dict
import os
import pandas as pd
import requests
from fetcher_base import DataFetcher, EVENTBRITE_SEARCH_URL, API_TIMEOUT, normalize_city
import logging

logger = logging.getLogger(__name__)


class EventbriteFetcher(DataFetcher):
    def __init__(self, output_file: str = 'eventbrite.csv'):
        super().__init__(output_file)
        self.token = os.getenv('EVENTBRITE_TOKEN')

        """Eventbrite API fetcher.

        Notes:
        - Uses `location.address` and a small `location.within` radius to find events.
        - If the API returns a non-JSON or 404, the fetcher will gracefully return
            zeroed metrics so downstream merging continues.
        """

    def _query_events(self, city: str, event_date: str) -> Dict:
        if not self.token:
            logger.warning('Eventbrite token missing; skipping Eventbrite enrichment')
            return {'eventbrite_event_count': 0, 'eventbrite_free_event_pct': 0.0, 'eventbrite_nonfree_count': 0}
        params = {
            'location.address': city,
            'location.within': '10mi',
            'start_date.range_start': f'{event_date}T00:00:00Z',
            'start_date.range_end': f'{event_date}T23:59:59Z',
            'expand': 'venue',
            'page': 1
        }
        headers = {'Authorization': f'Bearer {self.token}'}
        try:
            response = requests.get(EVENTBRITE_SEARCH_URL, params=params, headers={**headers, 'User-Agent': 'concert-pipeline/1.0'}, timeout=API_TIMEOUT)
            # Some Eventbrite accounts return HTML or 404 for unauthorized queries; handle safely
            if response.status_code == 404:
                logger.warning(f'Eventbrite returned 404 for {city} {event_date}')
                return {'eventbrite_event_count': 0, 'eventbrite_free_event_pct': 0.0, 'eventbrite_nonfree_count': 0}
            response.raise_for_status()
            # If the response isn't JSON, log a short snippet for debugging
            try:
                payload = response.json()
            except ValueError:
                txt = response.text.replace('\n', ' ')[:400]
                logger.warning(f'Eventbrite returned non-JSON for {city} {event_date}: {txt}')
                return {'eventbrite_event_count': 0, 'eventbrite_free_event_pct': 0.0, 'eventbrite_nonfree_count': 0}
            events = payload.get('events', [])
            if not events:
                return {'eventbrite_event_count': 0, 'eventbrite_free_event_pct': 0.0, 'eventbrite_nonfree_count': 0}

            free_count = sum(1 for event in events if event.get('is_free'))
            total = len(events)
            return {
                'eventbrite_event_count': total,
                'eventbrite_free_event_pct': (free_count / total) if total else 0.0,
                'eventbrite_nonfree_count': total - free_count
            }
        except Exception as exc:
            logger.warning(f'Eventbrite error for {city} on {event_date}: {exc}')
            return {'eventbrite_event_count': 0, 'eventbrite_free_event_pct': 0.0, 'eventbrite_nonfree_count': 0}

    def fetch(self, source_csv: str, start_date: str = None, end_date: str = None):
        # Accept artist column if present to improve search relevance
        usecols = ['city', 'event_date']
        try:
            sample = pd.read_csv(source_csv, nrows=1)
            if 'artist' in sample.columns:
                usecols.append('artist')
        except Exception:
            pass

        combos = []
        for chunk in pd.read_csv(source_csv, usecols=usecols, chunksize=50):
            chunk['city'] = chunk['city'].astype(str).apply(normalize_city)
            chunk['event_date'] = chunk['event_date'].astype(str)
            if 'artist' in chunk.columns:
                chunk['artist'] = chunk['artist'].astype(str).str.replace('/', ' ').str.strip()
            combos.append(chunk.drop_duplicates())
        if not combos:
            return pd.DataFrame()

        unique_rows = pd.concat(combos, ignore_index=True).drop_duplicates()
        for row in unique_rows.itertuples(index=False):
            artist = getattr(row, 'artist', None) if 'artist' in unique_rows.columns else None
            result = {'city': row.city, 'event_date': row.event_date}
            result.update(self._query_events(row.city, row.event_date))
            # If artist is present, try a secondary query with `q` parameter to find artist-specific events
            if artist:
                try:
                    params = {'q': artist, 'location.address': row.city, 'location.within': '25mi', 'start_date.range_start': f'{row.event_date}T00:00:00Z', 'start_date.range_end': f'{row.event_date}T23:59:59Z'}
                    resp = requests.get(EVENTBRITE_SEARCH_URL, params=params, headers={'Authorization': f'Bearer {self.token}', 'User-Agent': 'concert-pipeline/1.0'}, timeout=API_TIMEOUT)
                    if resp.status_code == 200:
                        try:
                            pl = resp.json()
                            events = pl.get('events', [])
                            if events:
                                free_count = sum(1 for event in events if event.get('is_free'))
                                total = len(events)
                                result.update({'eventbrite_event_count_artist': total, 'eventbrite_free_event_pct_artist': (free_count / total) if total else 0.0})
                        except ValueError:
                            pass
                except Exception:
                    pass
            self.data.append(result)
            self.success += 1
            self._delay()

        return pd.DataFrame(self.data)
