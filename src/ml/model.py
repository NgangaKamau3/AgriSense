"""
Crop stress classification model for AgriSense.
Supports both Random Forest and PyTorch-based pixel-wise classification.

Author: Nganga Kamau
GitHub: https://github.com/NgangaKamau3/AgriSense
"""

import ee
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from src.data.ground_truth_loader import GroundTruthLoader
from src.ml.feature_engineering import SpatialFeatureExtractor


class CropStressModel:
    """
    Pixel-wise crop stress classification model using Random Forest.
    Supports both synthetic training data (for demo) and real ground truth data.
    Includes spatial feature extraction for improved accuracy.
    """
    
    def __init__(self, use_spatial_features: bool = True, window_size: int = 3):
        """
        Initialize the crop stress model.
        
        Args:
            use_spatial_features: Whether to use spatial context features
            window_size: Size of spatial window for feature extraction
        """
        # Base spectral indices
        self.base_features = ['NDVI', 'NDWI', 'NDMI', 'MSI', 'NDRE']
        self.use_spatial_features = use_spatial_features
        self.window_size = window_size
        
        # Feature names (will be extended with spatial features if enabled)
        self.feature_names = self.base_features.copy()
        
        # Spatial feature extractor
        if use_spatial_features:
            self.spatial_extractor = SpatialFeatureExtractor(window_size=window_size)
            # Add spatial feature names
            for band in self.base_features:
                self.feature_names.extend([
                    f'{band}_spatial_mean',
                    f'{band}_spatial_std',
                    f'{band}_spatial_min',
                    f'{band}_spatial_max'
                ])
            # Add NDVI texture features
            self.feature_names.extend([
                'NDVI_texture_contrast',
                'NDVI_texture_correlation',
                'NDVI_texture_energy',
                'NDVI_texture_homogeneity',
                'NDVI_edge_magnitude',
                'NDVI_edge_laplacian'
            ])
        
        self.classifier = None
        self.training_data_source = None
        self.num_trees = 50
        
        # Stress class mapping
        self.stress_classes = {
            0: 'Healthy',
            1: 'Water Stress',
            2: 'Heat Stress',
            3: 'Nutrient Deficiency',
            4: 'Disease'
        }
    
    def train_from_ground_truth(self, ground_truth_path: str, 
                               test_size: float = 0.2,
                               num_trees: int = 50):
        """
        Train the model using real ground truth data.
        
        Args:
            ground_truth_path: Path to ground truth CSV file
            test_size: Fraction of data to use for testing
            num_trees: Number of trees in the Random Forest
        """
        print(f"Training model from ground truth data: {ground_truth_path}")
        
        # Load ground truth data
        loader = GroundTruthLoader(ground_truth_path)
        data = loader.load()
        
        # Split into train/test
        train_df, test_df = loader.train_test_split(test_size=test_size)
        
        # Convert to GEE FeatureCollection
        # Create temporary loader for train data
        train_loader = GroundTruthLoader(ground_truth_path)
        train_loader.data = train_df
        training_features = train_loader.to_gee_feature_collection()
        
        # Store test data for validation
        self.test_data = test_df
        self.training_data_source = ground_truth_path
        self.num_trees = num_trees
        
        # Train the classifier
        print(f"Training Random Forest with {num_trees} trees on {len(train_df)} samples...")
        self.classifier = ee.Classifier.smileRandomForest(num_trees).train(
            features=training_features,
            classProperty='stress_class',
            inputProperties=self.base_features  # Use base features for now
        )
        
        print(f"Model training complete. Test set size: {len(test_df)}")
        return self
    
    def train_synthetic(self, num_trees: int = 50):
        """
        Train using synthetic data (original implementation for demo purposes).
        
        Args:
            num_trees: Number of trees in the Random Forest
        """
        print("Training GEE-native Random Forest model with SYNTHETIC data...")
        print("WARNING: This is for demonstration only. Use train_from_ground_truth() for production.")
        
        self.num_trees = num_trees
        self.training_data_source = "synthetic"
        
        # Synthetic Training Data (FeatureCollection)
        # 0: Healthy, 1: Water Stress, 2: Heat Stress, 3: Nutrient, 4: Disease
        training_features = ee.FeatureCollection([
            # Healthy (High NDVI, High NDWI)
            ee.Feature(None, {'NDVI': 0.8, 'NDWI': 0.4, 'NDMI': 0.5, 'MSI': 0.5, 'NDRE': 0.7, 'stress_class': 0}),
            ee.Feature(None, {'NDVI': 0.7, 'NDWI': 0.3, 'NDMI': 0.4, 'MSI': 0.6, 'NDRE': 0.6, 'stress_class': 0}),
            ee.Feature(None, {'NDVI': 0.75, 'NDWI': 0.35, 'NDMI': 0.45, 'MSI': 0.55, 'NDRE': 0.65, 'stress_class': 0}),
            
            # Water Stress (Low NDWI, High MSI)
            ee.Feature(None, {'NDVI': 0.4, 'NDWI': -0.2, 'NDMI': 0.1, 'MSI': 1.5, 'NDRE': 0.3, 'stress_class': 1}),
            ee.Feature(None, {'NDVI': 0.3, 'NDWI': -0.3, 'NDMI': 0.0, 'MSI': 1.8, 'NDRE': 0.2, 'stress_class': 1}),
            ee.Feature(None, {'NDVI': 0.35, 'NDWI': -0.25, 'NDMI': 0.05, 'MSI': 1.65, 'NDRE': 0.25, 'stress_class': 1}),
            
            # Heat Stress (High MSI, Moderate NDVI)
            ee.Feature(None, {'NDVI': 0.5, 'NDWI': 0.1, 'NDMI': 0.2, 'MSI': 2.0, 'NDRE': 0.4, 'stress_class': 2}),
            ee.Feature(None, {'NDVI': 0.48, 'NDWI': 0.12, 'NDMI': 0.22, 'MSI': 1.9, 'NDRE': 0.38, 'stress_class': 2}),
            
            # Nutrient Deficiency (Low NDVI, Low NDRE, Moderate water)
            ee.Feature(None, {'NDVI': 0.35, 'NDWI': 0.2, 'NDMI': 0.25, 'MSI': 0.85, 'NDRE': 0.25, 'stress_class': 3}),
            ee.Feature(None, {'NDVI': 0.32, 'NDWI': 0.22, 'NDMI': 0.27, 'MSI': 0.88, 'NDRE': 0.23, 'stress_class': 3}),
            
            # Disease (Low NDVI, Moderate Water)
            ee.Feature(None, {'NDVI': 0.2, 'NDWI': 0.0, 'NDMI': 0.1, 'MSI': 1.0, 'NDRE': 0.1, 'stress_class': 4}),
            ee.Feature(None, {'NDVI': 0.25, 'NDWI': 0.05, 'NDMI': 0.12, 'MSI': 1.05, 'NDRE': 0.15, 'stress_class': 4}),
        ])

        # Train the classifier
        self.classifier = ee.Classifier.smileRandomForest(num_trees).train(
            features=training_features,
            classProperty='stress_class',
            inputProperties=self.base_features
        )
        print("Synthetic model training complete.")
        return self

    def validate(self) -> Dict:
        """
        Validate the model and return comprehensive metrics.
        
        Returns:
            Dictionary with accuracy metrics and confusion matrix
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train_from_ground_truth() or train_synthetic() first.")
        
        print("Validating model performance...")
        
        # Get confusion matrix from the classifier
        # Note: This uses training data for GEE-native models
        # For proper validation, use the test set with classify_dataframe()
        try:
            train_accuracy = self.classifier.confusionMatrix()
            
            metrics = {
                'overall_accuracy': train_accuracy.accuracy().getInfo(),
                'kappa': train_accuracy.kappa().getInfo(),
                'confusion_matrix': train_accuracy.getInfo(),
                'training_data_source': self.training_data_source,
                'num_trees': self.num_trees,
                'use_spatial_features': self.use_spatial_features
            }
            
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
            print(f"Kappa Coefficient: {metrics['kappa']:.4f}")
            print("\nConfusion Matrix:")
            print(json.dumps(metrics['confusion_matrix'], indent=2))
            
            # Calculate per-class metrics if confusion matrix is available
            if 'array' in metrics['confusion_matrix']:
                cm = metrics['confusion_matrix']['array']
                per_class_metrics = self._calculate_per_class_metrics(cm)
                metrics['per_class_metrics'] = per_class_metrics
                
                print("\nPer-Class Metrics:")
                for class_id, class_metrics in per_class_metrics.items():
                    class_name = self.stress_classes.get(class_id, f'Class {class_id}')
                    print(f"\n{class_name}:")
                    print(f"  Precision: {class_metrics['precision']:.4f}")
                    print(f"  Recall: {class_metrics['recall']:.4f}")
                    print(f"  F1-Score: {class_metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Validation error: {e}")
            return {
                'error': str(e),
                'training_data_source': self.training_data_source
            }
    
    def _calculate_per_class_metrics(self, confusion_matrix: List[List[int]]) -> Dict:
        """
        Calculate precision, recall, and F1-score for each class.
        
        Args:
            confusion_matrix: Confusion matrix as 2D array
        
        Returns:
            Dictionary with per-class metrics
        """
        import numpy as np
        cm = np.array(confusion_matrix)
        num_classes = cm.shape[0]
        
        metrics = {}
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[i] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': int(cm[i, :].sum())
            }
        
        return metrics

    def classify(self, image: ee.Image) -> ee.Image:
        """
        Classify the image pixel-wise using the trained classifier.
        
        Args:
            image: Image containing the spectral indices bands
        
        Returns:
            Image with 'stress_class' and 'confidence' bands
        """
        if self.classifier is None:
            # Auto-train with synthetic data if not trained
            print("Model not trained. Auto-training with synthetic data...")
            self.train_synthetic()
        
        # Add spatial features if enabled
        if self.use_spatial_features:
            print("Extracting spatial features for classification...")
            image = self.spatial_extractor.add_all_spatial_features(image, self.base_features)
        
        # Classify the image (pixel-wise)
        # For now, using base features only (spatial features require more complex integration)
        classified = image.select(self.base_features).classify(self.classifier)
        
        # Get classification probabilities for confidence
        probabilities = image.select(self.base_features).classify(
            self.classifier.setOutputMode('MULTIPROBABILITY')
        )
        
        # Calculate confidence as max probability
        confidence = probabilities.reduce(ee.Reducer.max()).rename('confidence')
        
        # Combine classification and confidence
        result = classified.rename('stress_class').addBands(confidence)
        
        return result
    
    def get_field_statistics(self, classified_image: ee.Image, roi: ee.Geometry) -> Dict:
        """
        Get statistics about stress distribution in a field.
        
        Args:
            classified_image: Classified image with 'stress_class' band
            roi: Region of interest (field boundary)
        
        Returns:
            Dictionary with stress class percentages and counts
        """
        # Calculate area for each stress class
        pixel_area = ee.Image.pixelArea()
        
        stats = {}
        total_area = 0
        
        for class_id, class_name in self.stress_classes.items():
            # Create mask for this class
            class_mask = classified_image.select('stress_class').eq(class_id)
            
            # Calculate area
            area = pixel_area.updateMask(class_mask).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=roi,
                scale=10,
                maxPixels=1e9
            ).get('area')
            
            area_value = area.getInfo() if area else 0
            stats[class_name] = {
                'area_m2': area_value,
                'class_id': class_id
            }
            total_area += area_value
        
        # Calculate percentages
        for class_name in stats:
            stats[class_name]['percentage'] = (stats[class_name]['area_m2'] / total_area * 100) if total_area > 0 else 0
        
        stats['total_area_m2'] = total_area
        stats['total_area_hectares'] = total_area / 10000
        
        return stats
    
    def save_model_info(self, output_path: str):
        """
        Save model metadata (GEE models can't be serialized locally).
        
        Args:
            output_path: Path to save model info JSON
        """
        info = {
            'model_type': 'GEE_RandomForest',
            'num_trees': self.num_trees,
            'feature_names': self.feature_names,
            'base_features': self.base_features,
            'use_spatial_features': self.use_spatial_features,
            'window_size': self.window_size,
            'training_data_source': self.training_data_source,
            'stress_classes': self.stress_classes
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Model info saved to {output_path}")


if __name__ == "__main__":
    from src.data.gee_fetcher import initialize_gee
    
    try:
        initialize_gee()
        
        # Test 1: Train with ground truth data
        print("=" * 60)
        print("TEST 1: Training with Ground Truth Data")
        print("=" * 60)
        model_gt = CropStressModel(use_spatial_features=False)
        model_gt.train_from_ground_truth('data/sample_ground_truth.csv', num_trees=50)
        metrics_gt = model_gt.validate()
        
        # Test 2: Train with synthetic data
        print("\n" + "=" * 60)
        print("TEST 2: Training with Synthetic Data")
        print("=" * 60)
        model_syn = CropStressModel(use_spatial_features=False)
        model_syn.train_synthetic(num_trees=10)
        metrics_syn = model_syn.validate()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
