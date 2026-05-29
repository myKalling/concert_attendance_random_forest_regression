"""
Concert Attendance Prediction Pipeline
Main orchestrator for the complete ML workflow.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from data_processor import prepare_data
from data_fetchers_oop import fetch_all_external_data
from dataset_merger import merge_all_datasets

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ConcertModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}

    def _build_preprocessor(self, numeric_cols, categorical_cols) -> ColumnTransformer:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        return ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    def train(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'rf',
              test_size: float = 0.2, use_cv: bool = True, cv_folds: int = 5) -> Dict:
        logger.info(f"Training model with {len(X)} samples")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        numeric_cols = self.X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = [
            c for c in self.X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            if c not in {'event_date'}
        ]
        logger.info(f"Numeric cols: {len(numeric_cols)}, categorical cols: {len(categorical_cols)}")

        preprocessor = self._build_preprocessor(numeric_cols, categorical_cols)
        estimator = GradientBoostingRegressor(n_estimators=120, random_state=self.random_state) if model_type == 'gb' else RandomForestRegressor(
            n_estimators=220, max_depth=12, random_state=self.random_state, n_jobs=-1
        )

        self.model = Pipeline(steps=[('preprocessor', preprocessor), ('estimator', estimator)])

        # If we have a sufficiently large training set, run a light randomized search
        do_tune = len(self.X_train) >= 50
        if do_tune and model_type == 'rf':
            logger.info('Performing RandomizedSearchCV tuning (light)')
            param_dist = {
                'estimator__n_estimators': [100, 200, 300],
                'estimator__max_depth': [6, 10, 16, None],
                'estimator__max_features': ['auto', 'sqrt', 'log2']
            }
            try:
                rs = RandomizedSearchCV(self.model, param_distributions=param_dist, n_iter=12, cv=3, scoring='r2', n_jobs=-1, random_state=self.random_state)
                rs.fit(self.X_train, self.y_train)
                logger.info(f'Best params: {rs.best_params_}')
                self.model = rs.best_estimator_
            except Exception as exc:
                logger.warning(f'Hyperparameter tuning failed: {exc}; falling back to default estimator')
                self.model.fit(self.X_train, self.y_train)
        else:
            self.model.fit(self.X_train, self.y_train)

        prediction = self.model.predict(self.X_test)
        self.metrics['test_r2'] = r2_score(self.y_test, prediction)
        self.metrics['test_rmse'] = np.sqrt(mean_squared_error(self.y_test, prediction))
        self.metrics['test_mae'] = mean_absolute_error(self.y_test, prediction)
        self.metrics['test_samples'] = len(self.X_test)
        logger.info(f"Test R²: {self.metrics['test_r2']:.4f}, RMSE: {self.metrics['test_rmse']:.4f}, MAE: {self.metrics['test_mae']:.4f}")

        if use_cv and len(self.X_train) >= max(cv_folds * 2, 20):
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv_folds, scoring='r2')
            self.metrics['cv_scores'] = cv_scores.tolist()
            self.metrics['cv_mean'] = cv_scores.mean()
            self.metrics['cv_std'] = cv_scores.std()
            logger.info(f"CV mean R²: {self.metrics['cv_mean']:.4f} ± {self.metrics['cv_std']:.4f}")
        else:
            logger.info("Skipping cross-validation due to small or unstable training set")

        return self.metrics

    def save_summary(self, output_file: str) -> None:
        with open(output_file, 'w') as f:
            f.write("Concert Attendance Prediction Model - Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Estimator: {self.model.named_steps['estimator'].__class__.__name__}\n")
            f.write(f"Train samples: {len(self.X_train)}\n")
            f.write(f"Test samples: {self.metrics.get('test_samples')}\n")
            f.write(f"Test R²: {self.metrics.get('test_r2')}\n")
            f.write(f"Test RMSE: {self.metrics.get('test_rmse')}\n")
            f.write(f"Test MAE: {self.metrics.get('test_mae')}\n")
            f.write(f"CV Mean R²: {self.metrics.get('cv_mean', 'N/A')}\n")
            f.write(f"CV Std R²: {self.metrics.get('cv_std', 'N/A')}\n")
        logger.info(f"Saved model summary to {output_file}")


class PipelineOrchestrator:
    def __init__(self, raw_csv: str, use_external_data: bool = True):
        self.raw_csv = raw_csv
        self.use_external_data = use_external_data
        self.results = {}

    def run(self, output_dir: str = '.') -> Dict:
        logger.info("=" * 60)
        logger.info("Concert Attendance Prediction Pipeline")
        logger.info(f"Input: {self.raw_csv} | Output: {output_dir}")
        logger.info("=" * 60)

        processed_csv = f"{output_dir}/concerts_processed.csv"
        df = prepare_data(self.raw_csv, processed_csv)
        self.results['processed_shape'] = df.shape

        if self.use_external_data:
            logger.info("Fetching external data...")
            fetch_all_external_data(processed_csv, output_dir)
            try:
                df = merge_all_datasets(processed_csv, output_dir)
                self.results['merged_shape'] = df.shape
            except Exception as exc:
                logger.warning(f"Merge failed, falling back to processed data: {exc}")
                df = pd.read_csv(processed_csv)
        else:
            df = pd.read_csv(processed_csv)

        if 'attendance_rate' not in df.columns:
            logger.error("attendance_rate missing; cannot train")
            return self.results

        # Prepare features: drop identifiers and date columns
        X = df.drop(columns=['event_id', 'attendance', 'attendance_rate', 'event_date'], errors='ignore')
        y = df['attendance_rate']
        mask = y.notna()
        X = X[mask].copy()
        y = y[mask]

        logger.info(f"Training set: {len(X)} rows, {X.shape[1]} features")

        # Basic feature selection and cleanup
        X = self._clean_features(X)

        model = ConcertModel()
        self.results['model_evaluation'] = model.train(X, y, model_type='rf')
        summary_path = f"{output_dir}/model_summary.txt"
        model.save_summary(summary_path)
        logger.info("Pipeline complete")
        return self.results

    def _clean_features(self, X):
        """Apply lightweight feature cleaning:
        - drop columns with >50% missing
        - drop constant columns
        - group rare categorical levels
        - coerce object columns to strings (for OneHotEncoder)
        """
        import numpy as np
        from sklearn.feature_selection import VarianceThreshold

        # Drop columns with too many missing values
        missing_pct = X.isnull().mean()
        drop_cols = missing_pct[missing_pct > 0.5].index.tolist()
        if drop_cols:
            logger.info(f"Dropping {len(drop_cols)} columns with >50% missing: {drop_cols}")
            X = X.drop(columns=drop_cols)

        # Drop constant columns
        constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
        if constant_cols:
            logger.info(f"Dropping constant columns: {constant_cols}")
            X = X.drop(columns=constant_cols)

        # Group rare categories for high-cardinality categorical features
        cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        for col in cat_cols:
            counts = X[col].fillna('___missing___').value_counts(dropna=False)
            if counts.size > 6:
                allowed = counts[counts >= 2].index.tolist()
                if len(allowed) < counts.size:
                    X[col] = X[col].where(X[col].isin(allowed), 'Other')
                    logger.info(f"Grouped rare levels in {col}: {counts.size} → {X[col].nunique(dropna=False)}")

        # Remove categorical columns that still have too many unique values for stable OHE
        cat_cols = [col for col in cat_cols if X[col].nunique(dropna=False) <= 12]
        dropped_high_cards = set(X.select_dtypes(include=['object', 'category', 'string']).columns) - set(cat_cols)
        if dropped_high_cards:
            logger.info(f"Dropping high-cardinality categorical cols: {sorted(dropped_high_cards)}")
        X_clean = pd.concat([X.select_dtypes(include=['number']), X[cat_cols].astype('object')], axis=1)
        X_clean = X_clean.replace({pd.NA: np.nan})
        return X_clean


def main():
    parser = argparse.ArgumentParser(description='Concert Attendance Prediction Pipeline')
    parser.add_argument('-i', '--input', default='concerts_raw.csv', help='Input CSV file')
    parser.add_argument('-o', '--output-dir', default='.', help='Output directory')
    parser.add_argument('--no-external-data', action='store_true', help='Skip external data fetch and merge')
    parser.add_argument('--quick', action='store_true', help='Quick mode: skip external fetch and merge')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    orchestrator = PipelineOrchestrator(args.input, use_external_data=not (args.no_external_data or args.quick))
    results = orchestrator.run(args.output_dir)
    logger.info(f"Results: {results}")


if __name__ == '__main__':
    main()
