import ee
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import json


class GroundTruthLoader:
    """
    Loads and validates ground truth training data for crop stress classification.
    Supports CSV and GeoJSON formats with spatial and temporal filtering.
    """
    
    # Expected columns for training data
    REQUIRED_COLUMNS = ['longitude', 'latitude', 'date', 'stress_class']
    FEATURE_COLUMNS = ['NDVI', 'NDWI', 'NDMI', 'MSI', 'NDRE']
    OPTIONAL_COLUMNS = ['crop_type', 'confidence', 'source', 'notes']
    
    # Stress class mapping
    STRESS_CLASSES = {
        0: 'Healthy',
        1: 'Water Stress',
        2: 'Heat Stress',
        3: 'Nutrient Deficiency',
        4: 'Disease'
    }
    
    def __init__(self, data_path: str):
        """
        Initialize the ground truth loader.
        
        Args:
            data_path: Path to CSV or GeoJSON file containing ground truth data
        """
        self.data_path = Path(data_path)
        self.data = None
        self._validate_path()
    
    def _validate_path(self):
        """Validate that the data file exists and has a supported format."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Ground truth data file not found: {self.data_path}")
        
        if self.data_path.suffix not in ['.csv', '.geojson', '.json']:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}. Use .csv or .geojson")
    
    def load(self) -> pd.DataFrame:
        """
        Load ground truth data from file.
        
        Returns:
            DataFrame with validated ground truth data
        """
        print(f"Loading ground truth data from {self.data_path}...")
        
        if self.data_path.suffix == '.csv':
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.suffix in ['.geojson', '.json']:
            self.data = self._load_geojson()
        
        self._validate_schema()
        self._preprocess()
        
        print(f"Loaded {len(self.data)} ground truth samples")
        print(f"Stress class distribution:\n{self.data['stress_class'].value_counts().sort_index()}")
        
        return self.data
    
    def _load_geojson(self) -> pd.DataFrame:
        """Load data from GeoJSON format."""
        with open(self.data_path, 'r') as f:
            geojson = json.load(f)
        
        records = []
        for feature in geojson['features']:
            props = feature['properties'].copy()
            coords = feature['geometry']['coordinates']
            props['longitude'] = coords[0]
            props['latitude'] = coords[1]
            records.append(props)
        
        return pd.DataFrame(records)
    
    def _validate_schema(self):
        """Validate that required columns are present."""
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check if we have either pre-computed features OR we'll need to fetch them
        has_features = all(col in self.data.columns for col in self.FEATURE_COLUMNS)
        if not has_features:
            print("Warning: Spectral indices not found in data. Will need to fetch from GEE.")
    
    def _preprocess(self):
        """Preprocess and clean the data."""
        # Convert date to datetime
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Validate coordinates
        if not self.data['longitude'].between(-180, 180).all():
            raise ValueError("Invalid longitude values (must be between -180 and 180)")
        if not self.data['latitude'].between(-90, 90).all():
            raise ValueError("Invalid latitude values (must be between -90 and 90)")
        
        # Validate stress classes
        valid_classes = set(self.STRESS_CLASSES.keys())
        invalid_classes = set(self.data['stress_class'].unique()) - valid_classes
        if invalid_classes:
            raise ValueError(f"Invalid stress classes found: {invalid_classes}. Valid: {valid_classes}")
        
        # Remove duplicates
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates(subset=['longitude', 'latitude', 'date'])
        if len(self.data) < initial_count:
            print(f"Removed {initial_count - len(self.data)} duplicate samples")
        
        # Sort by date
        self.data = self.data.sort_values('date').reset_index(drop=True)
    
    def filter_by_date(self, start_date: str, end_date: str) -> 'GroundTruthLoader':
        """
        Filter data by date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Self for method chaining
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        self.data = self.data[(self.data['date'] >= start) & (self.data['date'] <= end)]
        print(f"Filtered to {len(self.data)} samples between {start_date} and {end_date}")
        
        return self
    
    def filter_by_crop_type(self, crop_types: List[str]) -> 'GroundTruthLoader':
        """
        Filter data by crop type.
        
        Args:
            crop_types: List of crop types to include
        
        Returns:
            Self for method chaining
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        if 'crop_type' not in self.data.columns:
            print("Warning: crop_type column not found. Skipping filter.")
            return self
        
        self.data = self.data[self.data['crop_type'].isin(crop_types)]
        print(f"Filtered to {len(self.data)} samples for crop types: {crop_types}")
        
        return self
    
    def filter_by_region(self, bounds: Tuple[float, float, float, float]) -> 'GroundTruthLoader':
        """
        Filter data by geographic bounding box.
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
        
        Returns:
            Self for method chaining
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        min_lon, min_lat, max_lon, max_lat = bounds
        
        self.data = self.data[
            (self.data['longitude'] >= min_lon) &
            (self.data['longitude'] <= max_lon) &
            (self.data['latitude'] >= min_lat) &
            (self.data['latitude'] <= max_lat)
        ]
        print(f"Filtered to {len(self.data)} samples within bounds {bounds}")
        
        return self
    
    def to_gee_feature_collection(self) -> ee.FeatureCollection:
        """
        Convert ground truth data to GEE FeatureCollection.
        
        Returns:
            ee.FeatureCollection with training samples
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        features = []
        for _, row in self.data.iterrows():
            # Create properties dict
            properties = {
                'stress_class': int(row['stress_class']),
                'date': row['date'].strftime('%Y-%m-%d')
            }
            
            # Add spectral indices if available
            for col in self.FEATURE_COLUMNS:
                if col in row and pd.notna(row[col]):
                    properties[col] = float(row[col])
            
            # Add optional columns
            for col in self.OPTIONAL_COLUMNS:
                if col in row and pd.notna(row[col]):
                    properties[col] = str(row[col])
            
            # Create point geometry
            point = ee.Geometry.Point([row['longitude'], row['latitude']])
            
            # Create feature
            feature = ee.Feature(point, properties)
            features.append(feature)
        
        print(f"Created GEE FeatureCollection with {len(features)} samples")
        return ee.FeatureCollection(features)
    
    def train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        
        Returns:
            (train_df, test_df)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Stratified split to maintain class distribution
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state,
            stratify=self.data['stress_class']
        )
        
        print(f"Split: {len(train_df)} training samples, {len(test_df)} test samples")
        print(f"Training class distribution:\n{train_df['stress_class'].value_counts().sort_index()}")
        print(f"Test class distribution:\n{test_df['stress_class'].value_counts().sort_index()}")
        
        return train_df, test_df
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics about the ground truth data.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        stats = {
            'total_samples': len(self.data),
            'date_range': (self.data['date'].min(), self.data['date'].max()),
            'spatial_extent': {
                'min_lon': self.data['longitude'].min(),
                'max_lon': self.data['longitude'].max(),
                'min_lat': self.data['latitude'].min(),
                'max_lat': self.data['latitude'].max()
            },
            'class_distribution': self.data['stress_class'].value_counts().to_dict(),
            'class_names': {k: self.STRESS_CLASSES[k] for k in self.data['stress_class'].unique()}
        }
        
        if 'crop_type' in self.data.columns:
            stats['crop_types'] = self.data['crop_type'].value_counts().to_dict()
        
        return stats


if __name__ == "__main__":
    # Example usage
    from src.data.gee_fetcher import initialize_gee
    
    try:
        # Initialize GEE
        initialize_gee()
        
        # Load ground truth data
        loader = GroundTruthLoader('data/sample_ground_truth.csv')
        data = loader.load()
        
        # Get statistics
        stats = loader.get_summary_statistics()
        print("\nDataset Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        
        # Convert to GEE FeatureCollection
        fc = loader.to_gee_feature_collection()
        print(f"\nGEE FeatureCollection size: {fc.size().getInfo()}")
        
        # Train/test split
        train_df, test_df = loader.train_test_split(test_size=0.2)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
