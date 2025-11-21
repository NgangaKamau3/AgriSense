import ee
import numpy as np
from typing import List, Dict, Tuple, Optional


class SpatialFeatureExtractor:
    """
    Extracts spatial context features from satellite imagery for pixel-wise classification.
    Includes neighboring pixel values, texture features, and temporal statistics.
    """
    
    def __init__(self, window_size: int = 3):
        """
        Initialize the feature extractor.
        
        Args:
            window_size: Size of the spatial window for context extraction (3, 5, 7, etc.)
        """
        self.window_size = window_size
        self.radius = window_size // 2
    
    def add_spatial_context(self, image: ee.Image, band_names: List[str]) -> ee.Image:
        """
        Add spatial context features by including neighboring pixel values.
        For each band, creates additional bands with mean, std, min, max in the window.
        
        Args:
            image: Input image with spectral indices
            band_names: List of band names to extract spatial context for
        
        Returns:
            Image with additional spatial context bands
        """
        result = image
        
        # Define kernel for neighborhood operations
        kernel = ee.Kernel.square(radius=self.radius, units='pixels')
        
        for band in band_names:
            band_image = image.select(band)
            
            # Calculate neighborhood statistics
            mean = band_image.reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=kernel
            ).rename(f'{band}_spatial_mean')
            
            std = band_image.reduceNeighborhood(
                reducer=ee.Reducer.stdDev(),
                kernel=kernel
            ).rename(f'{band}_spatial_std')
            
            min_val = band_image.reduceNeighborhood(
                reducer=ee.Reducer.min(),
                kernel=kernel
            ).rename(f'{band}_spatial_min')
            
            max_val = band_image.reduceNeighborhood(
                reducer=ee.Reducer.max(),
                kernel=kernel
            ).rename(f'{band}_spatial_max')
            
            # Add to result
            result = result.addBands([mean, std, min_val, max_val])
        
        return result
    
    def add_texture_features(self, image: ee.Image, band_name: str = 'NDVI') -> ee.Image:
        """
        Add texture features using GLCM (Gray-Level Co-occurrence Matrix).
        Computes contrast, correlation, energy, and homogeneity.
        
        Args:
            image: Input image
            band_name: Band to compute texture features for (typically NDVI)
        
        Returns:
            Image with additional texture feature bands
        """
        # Select the band and scale to 0-255 for GLCM
        band = image.select(band_name)
        
        # Normalize to 0-255 range (GLCM requirement)
        # Assuming NDVI range is -1 to 1
        normalized = band.add(1).multiply(127.5).byte()
        
        # Compute GLCM
        glcm = normalized.glcmTexture(size=self.window_size)
        
        # Select key texture features
        contrast = glcm.select(f'{band_name}_contrast').rename(f'{band_name}_texture_contrast')
        correlation = glcm.select(f'{band_name}_corr').rename(f'{band_name}_texture_correlation')
        
        # ASM (Angular Second Moment) is energy squared
        asm = glcm.select(f'{band_name}_asm').rename(f'{band_name}_texture_energy')
        
        # IDM (Inverse Difference Moment) is homogeneity
        idm = glcm.select(f'{band_name}_idm').rename(f'{band_name}_texture_homogeneity')
        
        return image.addBands([contrast, correlation, asm, idm])
    
    def add_edge_features(self, image: ee.Image, band_name: str = 'NDVI') -> ee.Image:
        """
        Add edge detection features using gradient-based methods.
        
        Args:
            image: Input image
            band_name: Band to compute edge features for
        
        Returns:
            Image with edge feature bands
        """
        band = image.select(band_name)
        
        # Sobel edge detection
        sobel = band.convolve(ee.Kernel.sobel())
        sobel = sobel.rename(f'{band_name}_edge_magnitude')
        
        # Laplacian edge detection
        laplacian = band.convolve(ee.Kernel.laplacian8())
        laplacian = laplacian.rename(f'{band_name}_edge_laplacian')
        
        return image.addBands([sobel, laplacian])
    
    def add_all_spatial_features(self, image: ee.Image, 
                                 spectral_bands: List[str] = None) -> ee.Image:
        """
        Add all spatial features: context, texture, and edges.
        
        Args:
            image: Input image with spectral indices
            spectral_bands: List of spectral index bands (default: standard indices)
        
        Returns:
            Image with all spatial features added
        """
        if spectral_bands is None:
            spectral_bands = ['NDVI', 'NDWI', 'NDMI', 'MSI', 'NDRE']
        
        # Add spatial context for all bands
        result = self.add_spatial_context(image, spectral_bands)
        
        # Add texture features for NDVI (most important for vegetation)
        result = self.add_texture_features(result, 'NDVI')
        
        # Add edge features for NDVI
        result = self.add_edge_features(result, 'NDVI')
        
        return result


class TemporalFeatureExtractor:
    """
    Extracts temporal features from time series of satellite imagery.
    Computes statistics and trends over time.
    """
    
    def __init__(self):
        """Initialize the temporal feature extractor."""
        pass
    
    def add_temporal_statistics(self, collection: ee.ImageCollection, 
                                band_names: List[str]) -> ee.Image:
        """
        Compute temporal statistics from an image collection.
        
        Args:
            collection: ImageCollection with time series data
            band_names: List of bands to compute statistics for
        
        Returns:
            Image with temporal statistics bands
        """
        # Compute statistics
        mean = collection.select(band_names).mean().rename(
            [f'{b}_temporal_mean' for b in band_names]
        )
        
        std = collection.select(band_names).reduce(ee.Reducer.stdDev()).rename(
            [f'{b}_temporal_std' for b in band_names]
        )
        
        min_val = collection.select(band_names).min().rename(
            [f'{b}_temporal_min' for b in band_names]
        )
        
        max_val = collection.select(band_names).max().rename(
            [f'{b}_temporal_max' for b in band_names]
        )
        
        # Combine all statistics
        result = mean.addBands([std, min_val, max_val])
        
        return result
    
    def add_temporal_trend(self, collection: ee.ImageCollection, 
                          band_name: str = 'NDVI') -> ee.Image:
        """
        Compute temporal trend (slope) for a band using linear regression.
        
        Args:
            collection: ImageCollection with time series data
            band_name: Band to compute trend for
        
        Returns:
            Image with trend band
        """
        # Add time band (days since first image)
        def add_time_band(image):
            time = image.date().difference(ee.Date(collection.first().date()), 'day')
            return image.addBands(ee.Image(time).rename('time').float())
        
        collection_with_time = collection.map(add_time_band)
        
        # Linear regression: band ~ time
        linear_fit = collection_with_time.select(['time', band_name]).reduce(
            ee.Reducer.linearFit()
        )
        
        # Extract slope (trend)
        trend = linear_fit.select('scale').rename(f'{band_name}_temporal_trend')
        
        return trend
    
    def add_all_temporal_features(self, collection: ee.ImageCollection,
                                  spectral_bands: List[str] = None) -> ee.Image:
        """
        Add all temporal features: statistics and trends.
        
        Args:
            collection: ImageCollection with time series data
            spectral_bands: List of spectral index bands
        
        Returns:
            Image with all temporal features
        """
        if spectral_bands is None:
            spectral_bands = ['NDVI', 'NDWI', 'NDMI', 'MSI', 'NDRE']
        
        # Add temporal statistics
        result = self.add_temporal_statistics(collection, spectral_bands)
        
        # Add trend for NDVI
        trend = self.add_temporal_trend(collection, 'NDVI')
        result = result.addBands(trend)
        
        return result


def extract_training_samples(image: ee.Image, 
                            ground_truth_fc: ee.FeatureCollection,
                            feature_bands: List[str],
                            scale: int = 10) -> ee.FeatureCollection:
    """
    Extract training samples from an image at ground truth locations.
    
    Args:
        image: Image with all feature bands
        ground_truth_fc: FeatureCollection with ground truth points
        feature_bands: List of feature band names to extract
        scale: Spatial resolution in meters (10m for Sentinel-2)
    
    Returns:
        FeatureCollection with extracted feature values and labels
    """
    # Sample the image at ground truth points
    samples = image.select(feature_bands).sampleRegions(
        collection=ground_truth_fc,
        properties=['stress_class'],
        scale=scale,
        geometries=True
    )
    
    return samples


def normalize_features(image: ee.Image, 
                      feature_bands: List[str],
                      stats: Optional[Dict[str, Tuple[float, float]]] = None) -> ee.Image:
    """
    Normalize features to [0, 1] range or standardize using mean/std.
    
    Args:
        image: Image with feature bands
        feature_bands: List of feature bands to normalize
        stats: Optional dict with {band_name: (mean, std)} for standardization
    
    Returns:
        Image with normalized features
    """
    result = image
    
    if stats is None:
        # Simple min-max normalization (assumes typical ranges)
        # This is a simplified version; in practice, compute from training data
        for band in feature_bands:
            # For most indices, range is approximately -1 to 1 or 0 to 2
            # Normalize to 0-1
            normalized = image.select(band).unitScale(-1, 2).clamp(0, 1)
            result = result.addBands(normalized.rename(f'{band}_normalized'), overwrite=True)
    else:
        # Standardization using provided statistics
        for band in feature_bands:
            if band in stats:
                mean, std = stats[band]
                standardized = image.select(band).subtract(mean).divide(std)
                result = result.addBands(standardized.rename(f'{band}_standardized'), overwrite=True)
    
    return result


if __name__ == "__main__":
    # Example usage
    from src.data.gee_fetcher import initialize_gee, get_temporal_composite
    from src.processing.indices import add_all_indices
    
    try:
        # Initialize GEE
        initialize_gee()
        
        # Define a test ROI
        roi = ee.Geometry.Polygon([
            [[36.8, -1.3], [36.9, -1.3], [36.9, -1.2], [36.8, -1.2], [36.8, -1.3]]
        ])
        
        # Get composite image
        print("Fetching composite image...")
        image = get_temporal_composite(roi, days=30)
        
        # Add spectral indices
        print("Adding spectral indices...")
        image = add_all_indices(image)
        
        # Extract spatial features
        print("Extracting spatial features...")
        spatial_extractor = SpatialFeatureExtractor(window_size=3)
        image_with_spatial = spatial_extractor.add_all_spatial_features(image)
        
        # Get band names
        band_names = image_with_spatial.bandNames().getInfo()
        print(f"\nTotal bands after spatial feature extraction: {len(band_names)}")
        print(f"Band names: {band_names[:10]}... (showing first 10)")
        
        print("\nFeature engineering test successful!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
