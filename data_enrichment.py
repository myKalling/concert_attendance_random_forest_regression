"""
Utility to augment the dataset with historical Ticketmaster events.
This will fetch past events for artists present in `concerts_processed.csv` and
write an `concerts_augmented.csv` file containing found events. These rows
may lack ground-truth `attendance` values and are intended for manual labeling
or downstream semi-supervised workflows.

Usage:
    python data_enrichment.py --input concerts_processed.csv --output concerts_augmented.csv --max 200
"""
import argparse
import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import requests
from fetcher_base import TICKETMASTER_EVENTS_URL, API_TIMEOUT, normalize_city

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


def augment_with_ticketmaster_history(input_csv: str, output_csv: str, max_rows: int = 200):
    api_key = os.getenv('TICKETMASTER_API_KEY')
    if not api_key:
        logger.error('TICKETMASTER_API_KEY missing; cannot augment')
        return

    df = pd.read_csv(input_csv)
    artists = df['artist'].dropna().astype(str).unique() if 'artist' in df.columns else []
    collected = []
    cutoff = datetime.utcnow().date()
    start = cutoff - timedelta(days=365 * 3)

    for artist in artists:
        if len(collected) >= max_rows:
            break
        params = {'apikey': api_key, 'keyword': artist, 'startDateTime': f'{start}T00:00:00Z', 'endDateTime': f'{cutoff}T23:59:59Z', 'size': 50}
        try:
            resp = requests.get(TICKETMASTER_EVENTS_URL, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            events = resp.json().get('_embedded', {}).get('events', [])
            for ev in events:
                # Map fields into a compatible processed row; attendance is unknown
                name = ev.get('name')
                dates = ev.get('dates', {}).get('start', {})
                date = dates.get('localDate')
                venues = ev.get('_embedded', {}).get('venues', [{}])
                city = venues[0].get('city', {}).get('name') if venues else None
                priceRanges = ev.get('priceRanges') or []
                min_price = priceRanges[0].get('min') if priceRanges else None
                max_price = priceRanges[0].get('max') if priceRanges else None
                row = {
                    'artist': artist,
                    'event_name': name,
                    'city': normalize_city(city) if city else None,
                    'event_date': date,
                    'ticket_price': (min_price + max_price) / 2 if min_price and max_price else (min_price or max_price),
                    'source': 'ticketmaster_history'
                }
                collected.append(row)
                if len(collected) >= max_rows:
                    break
        except Exception as exc:
            logger.warning(f'Historical fetch failed for {artist}: {exc}')

    if collected:
        out = pd.DataFrame(collected)
        out.to_csv(output_csv, index=False)
        logger.info(f'Wrote {len(out)} augmented rows to {output_csv}')
    else:
        logger.info('No historical events found')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='concerts_processed.csv')
    parser.add_argument('--output', default='concerts_augmented.csv')
    parser.add_argument('--max', type=int, default=200)
    args = parser.parse_args()
    augment_with_ticketmaster_history(args.input, args.output, args.max)
