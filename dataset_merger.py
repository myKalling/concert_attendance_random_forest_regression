"""
Dataset Merger: merge cleaned concert data with external API datasets.
"""
import pandas as pd
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
        self.df['city'] = self.df['city'].astype(str).str.strip()
        self.df['event_date'] = pd.to_datetime(self.df['event_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        self._log(f"Loaded primary dataset: {self.df.shape}")
        return self.df

    def _merge_on(self, external_df: pd.DataFrame, on: list, name: str):
        if self.df is None:
            raise ValueError('Primary dataset not loaded')
        for col in on:
            if col not in external_df.columns:
                logger.warning(f"Skipping {name} merge: missing {col}")
                return
        external_df = external_df.copy()
        if 'city' in external_df.columns:
            external_df['city'] = external_df['city'].astype(str).str.strip()
        if 'event_date' in external_df.columns:
            external_df['event_date'] = pd.to_datetime(external_df['event_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        external_df = external_df.drop_duplicates(subset=on)
        before = self.df.shape[1]
        self.df = self.df.merge(external_df, on=on, how='left')
        self._log(f"Merged {name} data: +{self.df.shape[1] - before} columns")

    def merge_spotify(self, spotify_df: pd.DataFrame) -> None:
        if spotify_df.empty:
            logger.warning('Skipping Spotify merge: no data')
            return
        self._merge_on(spotify_df, ['genre'], 'Spotify')

    def merge_bandsintown(self, bands_df: pd.DataFrame) -> None:
        if bands_df.empty:
            logger.warning('Skipping Bandsintown merge: no data')
            return
        self._merge_on(bands_df, ['genre'], 'Bandsintown')

    def merge_spotify_artists(self, spotify_artists_df: pd.DataFrame) -> None:
        if spotify_artists_df.empty:
            logger.warning('Skipping Spotify artist merge: no data')
            return
        self._merge_on(spotify_artists_df, ['genre'], 'SpotifyArtists')

    def merge_musicbrainz(self, musicbrainz_df: pd.DataFrame) -> None:
        if musicbrainz_df.empty:
            logger.warning('Skipping MusicBrainz merge: no data')
            return
        self._merge_on(musicbrainz_df, ['genre'], 'MusicBrainz')

    def merge_weather(self, weather_df: pd.DataFrame) -> None:
        if weather_df.empty:
            logger.warning('Skipping weather merge: no data')
            return
        self._merge_on(weather_df, ['city', 'event_date'], 'Weather')

    def merge_ticketmaster(self, ticketmaster_df: pd.DataFrame) -> None:
        if ticketmaster_df.empty:
            logger.warning('Skipping Ticketmaster merge: no data')
            return
        self._merge_on(ticketmaster_df, ['city', 'event_date'], 'Ticketmaster')

    def merge_eventbrite(self, eventbrite_df: pd.DataFrame) -> None:
        if eventbrite_df.empty:
            logger.warning('Skipping Eventbrite merge: no data')
            return
        self._merge_on(eventbrite_df, ['city', 'event_date'], 'Eventbrite')

    def merge_venue_stats(self, venues_df: pd.DataFrame) -> None:
        if venues_df.empty:
            logger.warning('Skipping venue merge: no data')
            return
        self._merge_on(venues_df, ['city'], 'VenueStats')

    def merge_demographics(self, demographics_df: pd.DataFrame) -> None:
        if demographics_df.empty:
            logger.warning('Skipping demographics merge: no data')
            return
        self._merge_on(demographics_df, ['city'], 'Demographics')

    def merge_setlistfm(self, setlist_df: pd.DataFrame, spotify_df: pd.DataFrame = None) -> None:
        if setlist_df.empty:
            logger.warning('Skipping setlist merge: no data')
            return
        if spotify_df is None or spotify_df.empty:
            logger.warning('Skipping setlist merge: spotify metadata required')
            return
        if 'genre' not in self.df.columns or 'spotify_top_artist' not in spotify_df.columns:
            logger.warning('Skipping setlist merge: missing genre or spotify artist data')
            return
        # Map genre to top artist from Spotify and attach setlist metrics
        mapping = spotify_df[['genre', 'spotify_top_artist']].dropna().drop_duplicates()
        setlist_df = setlist_df.copy()
        setlist_df['artist'] = setlist_df['artist'].astype(str).str.strip()
        cols = ['artist', 'setlist_recent_count']
        if 'artist' not in setlist_df.columns:
            logger.warning('Skipping setlist merge: artist column missing')
            return
        joined = mapping.merge(setlist_df[cols], left_on='spotify_top_artist', right_on='artist', how='left').drop(columns=['artist'])
        self._merge_on(joined, ['genre'], 'SetlistFM')

    def merge_ticket_availability(self, availability_df: pd.DataFrame) -> None:
        if availability_df.empty:
            logger.warning('Skipping ticket availability merge: no data')
            return
        av = availability_df.copy()
        if 'city' in av.columns:
            av['city'] = av['city'].astype(str).str.strip()
        if 'event_date' in av.columns:
            av['event_date'] = pd.to_datetime(av['event_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        if 'ticket_est_count' in av.columns:
            av = av.groupby(['city', 'event_date'], as_index=False).agg({
                'has_tickets': 'max',
                'ticket_est_count': 'sum'
            })
        self._merge_on(av, ['city', 'event_date'], 'TicketAvailability')

    def merge_geodata(self, geodata_df: pd.DataFrame) -> None:
        if geodata_df.empty:
            logger.warning('Skipping geodata merge: no data')
            return
        self._merge_on(geodata_df, ['city'], 'GeoData')

    def merge_concerts_augmented(self, augmented_df: pd.DataFrame) -> None:
        """Merge augmented historical events (may lack attendance). Uses city+event_date match."""
        if augmented_df.empty:
            logger.warning('Skipping augmented concerts merge: no data')
            return
        aug = augmented_df.copy()
        # Normalize keys
        if 'city' in aug.columns:
            aug['city'] = aug['city'].astype(str).str.strip()
        if 'event_date' in aug.columns:
            aug['event_date'] = pd.to_datetime(aug['event_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        # Drop duplicate keys to avoid explosion
        aug = aug.drop_duplicates(subset=['city', 'event_date'])
        # Prefix augmented columns to avoid name collisions
        key_cols = ['city', 'event_date']
        other_cols = [c for c in aug.columns if c not in key_cols]
        aug = aug.rename(columns={c: f'aug_{c}' for c in other_cols})
        self._merge_on(aug, ['city', 'event_date'], 'AugmentedConcerts')

    def engineer_post_merge(self) -> None:
        if self.df is None:
            raise ValueError('Primary dataset not loaded')
        if 'median_income' in self.df.columns and 'venue_capacity' in self.df.columns:
            self.df['income_per_capacity'] = self.df['median_income'] / (self.df['venue_capacity'] + 1)
        if 'ticketmaster_event_count' in self.df.columns and 'competing_events' in self.df.columns:
            self.df['ticket_demand_ratio'] = self.df['ticketmaster_event_count'] / (self.df['competing_events'] + 1)
        if 'weather_precipitation' in self.df.columns and 'is_weekend' in self.df.columns:
            self.df['rain_weekend'] = ((self.df['weather_precipitation'] > 0) & (self.df['is_weekend'] == 1)).astype(int)
        logger.info('Created post-merge derived features')

    def handle_missing(self) -> None:
        if self.df is None:
            raise ValueError('Primary dataset not loaded')
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        self.df['eventbrite_free_event_pct'] = self.df.get('eventbrite_free_event_pct', pd.Series(dtype=float)).fillna(0.0)
        self.df['ticketmaster_event_count'] = self.df.get('ticketmaster_event_count', pd.Series(dtype=float)).fillna(0.0)
        self.df['eventbrite_event_count'] = self.df.get('eventbrite_event_count', pd.Series(dtype=float)).fillna(0.0)
        self.df['venue_count'] = self.df.get('venue_count', pd.Series(dtype=float)).fillna(0.0)
        logger.info('Applied missing value strategy to merged dataset')

    def get_stats(self) -> dict:
        if self.df is None:
            return {}
        return {
            'shape': self.df.shape,
            'missing_pct': round(self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100, 2)
        }


def merge_all_datasets(concerts_csv: str, data_dir: str = '.') -> pd.DataFrame:
    logger.info('Starting dataset merge workflow...')
    merger = DatasetMerger()
    merger.load_primary(concerts_csv)

    sources = [
        ('spotify', merger.merge_spotify),
        ('spotify_artists', merger.merge_spotify_artists),
        ('bandsintown', merger.merge_bandsintown),
        ('musicbrainz', merger.merge_musicbrainz),
        ('weather', merger.merge_weather),
        ('ticketmaster', merger.merge_ticketmaster),
        ('eventbrite', merger.merge_eventbrite),
        ('venues', merger.merge_venue_stats),
        ('demographics', merger.merge_demographics)
    ]
    additional_sources = [
        ('setlistfm', merger.merge_setlistfm),
        ('ticket_availability', merger.merge_ticket_availability),
        ('geodata', merger.merge_geodata),
        ('concerts_augmented', merger.merge_concerts_augmented)
    ]

    for name, merge_method in sources:
        try:
            external_path = f'{data_dir}/{name}.csv'
            if pd.io.common.file_exists(external_path):
                external_df = pd.read_csv(external_path)
                merge_method(external_df)
            else:
                logger.warning(f'{name}.csv not found, skipping merge')
        except Exception as exc:
            logger.error(f'Error merging {name}: {exc}')

    # Merge supplementary sources that may require additional context
    for name, merge_method in additional_sources:
        try:
            external_path = f'{data_dir}/{name}.csv'
            if pd.io.common.file_exists(external_path):
                external_df = pd.read_csv(external_path)
                if name == 'setlistfm':
                    spotify_df = pd.read_csv(f'{data_dir}/spotify.csv') if pd.io.common.file_exists(f'{data_dir}/spotify.csv') else pd.DataFrame()
                    merge_method(external_df, spotify_df)
                else:
                    merge_method(external_df)
            else:
                logger.warning(f'{name}.csv not found, skipping merge')
        except Exception as exc:
            logger.error(f'Error merging {name}: {exc}')

    merger.handle_missing()
    merger.engineer_post_merge()
    output_path = f'{data_dir}/concerts_merged.csv'
    merger.df.to_csv(output_path, index=False)
    logger.info(f'Saved merged dataset to {output_path}: {merger.df.shape}')
    return merger.df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    merge_all_datasets('concerts_processed.csv', '.')
