"""
GEE deployment wrapper for PyTorch-trained models.
Provides seamless integration with Google Earth Engine.

Author: Nganga Kamau
GitHub: https://github.com/NgangaKamau3/AgriSense
"""

import ee
from typing import Optional, Dict, List
import json
from pathlib import Path


class PyTorchGEEModel:
    """
    Wrapper for PyTorch-trained model deployed on GEE via AI Platform.
    Provides the same interface as CropStressModel for easy switching.
    """
    
    def __init__(self, 
                 project_name: str,
                 model_name: str = 'stress_classifier',
                 version: str = 'v1',
                 input_tile_size: int = 256,
                 input_overlap: int = 32):
        """
        Initialize connection to deployed PyTorch model on GEE.
        
        Args:
            project_name: Google Cloud project name
            model_name: Model name on AI Platform
            version: Model version
            input_tile_size: Size of input tiles for processing
            input_overlap: Overlap between tiles to avoid edge artifacts
        """
        self.project_name = project_name
        self.model_name = model_name
        self.version = version
        self.input_tile_size = input_tile_size
        self.input_overlap = input_overlap
        
        # Stress class mapping (same as Random Forest)
        self.stress_classes = {
            0: 'Healthy',
            1: 'Water Stress',
            2: 'Heat Stress',
            3: 'Nutrient Deficiency',
            4: 'Disease'
        }
        
        # Feature names
        self.feature_names = ['NDVI', 'NDWI', 'NDMI', 'MSI', 'NDRE']
        
        # Initialize model connection
        self._init_model()
    
    def _init_model(self):
        """Initialize connection to AI Platform model."""
        try:
            # Connect to deployed model
            self.model = ee.Model.fromAiPlatformPredictor(
                projectName=self.project_name,
                modelName=self.model_name,
                version=self.version,
                inputTileSize=[self.input_tile_size, self.input_tile_size],
                inputOverlapSize=[self.input_overlap, self.input_overlap],
                proj=ee.Projection('EPSG:3857').atScale(10),
                fixInputProj=True,
                outputBands={
                    'output': {
                        'type': ee.PixelType.float(),
                        'dimensions': 1
                    }
                }
            )
            print(f"✓ Connected to model: {self.model_name} (version {self.version})")
            
        except Exception as e:
            print(f"Warning: Could not connect to AI Platform model: {e}")
            print("Make sure the model is deployed to AI Platform.")
            print("See export_to_tensorflow.py for deployment instructions.")
            self.model = None
    
    def classify(self, image: ee.Image) -> ee.Image:
        """
        Classify image using deployed PyTorch model.
        
        Args:
            image: ee.Image with spectral indices (NDVI, NDWI, NDMI, MSI, NDRE)
        
        Returns:
            ee.Image with 'stress_class' and 'confidence' bands
        """
        if self.model is None:
            raise ValueError("Model not initialized. Check AI Platform deployment.")
        
        # Prepare input (select and normalize bands)
        input_image = image.select(self.feature_names)
        
        # Normalize to [-1, 1] range (model expects this)
        input_image = input_image.divide(2.0)  # Spectral indices are typically -1 to 1
        
        # Run prediction on GEE servers
        prediction = self.model.predictImage(input_image)
        
        # Get class probabilities (model outputs logits)
        logits = prediction.select('output')
        
        # Apply softmax to get probabilities
        # Note: GEE doesn't have native softmax, so we approximate
        # For deployment, the model should output probabilities directly
        
        # Get predicted class (argmax)
        stress_class = logits.arrayArgmax().arrayGet([0]).rename('stress_class')
        
        # Get confidence (max probability)
        # For simplicity, we use the max logit as a proxy
        confidence = logits.arrayReduce(ee.Reducer.max(), [0]).arrayGet([0]).rename('confidence')
        
        return stress_class.addBands(confidence)
    
    def get_field_statistics(self, classified_image: ee.Image, roi: ee.Geometry) -> Dict:
        """
        Get statistics about stress distribution in a field.
        Same interface as CropStressModel for compatibility.
        
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
            stats[class_name]['percentage'] = (
                stats[class_name]['area_m2'] / total_area * 100
            ) if total_area > 0 else 0
        
        stats['total_area_m2'] = total_area
        stats['total_area_hectares'] = total_area / 10000
        
        return stats
    
    def validate(self) -> Dict:
        """
        Validate model (placeholder for compatibility).
        
        Returns:
            Dictionary with model info
        """
        return {
            'model_type': 'PyTorch_EfficientStressNet',
            'deployment': 'GEE_AI_Platform',
            'project': self.project_name,
            'model_name': self.model_name,
            'version': self.version,
            'status': 'deployed' if self.model is not None else 'not_deployed'
        }


class HybridModel:
    """
    Hybrid model that can switch between PyTorch and Random Forest.
    Provides automatic fallback if PyTorch model is unavailable.
    """
    
    def __init__(self,
                 use_pytorch: bool = True,
                 project_name: Optional[str] = None,
                 pytorch_model_name: str = 'stress_classifier',
                 pytorch_version: str = 'v1'):
        """
        Initialize hybrid model.
        
        Args:
            use_pytorch: Whether to use PyTorch model (if available)
            project_name: GCP project name for PyTorch model
            pytorch_model_name: PyTorch model name on AI Platform
            pytorch_version: PyTorch model version
        """
        self.use_pytorch = use_pytorch
        self.pytorch_model = None
        self.rf_model = None
        
        # Try to initialize PyTorch model
        if use_pytorch and project_name:
            try:
                from src.ml.gee_pytorch_model import PyTorchGEEModel
                self.pytorch_model = PyTorchGEEModel(
                    project_name=project_name,
                    model_name=pytorch_model_name,
                    version=pytorch_version
                )
                print("✓ Using PyTorch model")
            except Exception as e:
                print(f"Warning: Could not load PyTorch model: {e}")
                print("Falling back to Random Forest")
                self.use_pytorch = False
        
        # Initialize Random Forest as fallback
        if not self.use_pytorch or self.pytorch_model is None:
            from src.ml.model import CropStressModel
            self.rf_model = CropStressModel(use_spatial_features=False)
            print("✓ Using Random Forest model")
    
    def classify(self, image: ee.Image) -> ee.Image:
        """
        Classify image using available model.
        
        Args:
            image: ee.Image with spectral indices
        
        Returns:
            ee.Image with classification
        """
        if self.pytorch_model is not None:
            return self.pytorch_model.classify(image)
        else:
            return self.rf_model.classify(image)
    
    def get_field_statistics(self, classified_image: ee.Image, roi: ee.Geometry) -> Dict:
        """Get field statistics using available model."""
        if self.pytorch_model is not None:
            return self.pytorch_model.get_field_statistics(classified_image, roi)
        else:
            return self.rf_model.get_field_statistics(classified_image, roi)
    
    def get_model_info(self) -> Dict:
        """Get information about active model."""
        if self.pytorch_model is not None:
            return {
                'active_model': 'PyTorch',
                'details': self.pytorch_model.validate()
            }
        else:
            return {
                'active_model': 'RandomForest',
                'details': {
                    'training_source': getattr(self.rf_model, 'training_data_source', 'unknown'),
                    'num_trees': getattr(self.rf_model, 'num_trees', 'unknown')
                }
            }


if __name__ == "__main__":
    from src.data.gee_fetcher import initialize_gee
    
    try:
        # Initialize GEE
        initialize_gee()
        
        # Test PyTorch model connection
        print("Testing PyTorch GEE Model...")
        print("=" * 60)
        
        # Note: Replace with your actual GCP project name
        project_name = "your-gcp-project"
        
        model = PyTorchGEEModel(
            project_name=project_name,
            model_name='stress_classifier',
            version='v1'
        )
        
        # Validate
        info = model.validate()
        print("\nModel Info:")
        print(json.dumps(info, indent=2))
        
        print("\n✓ PyTorch GEE Model test successful!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure to:")
        print("1. Deploy model to AI Platform (see export_to_tensorflow.py)")
        print("2. Update project_name with your GCP project")
