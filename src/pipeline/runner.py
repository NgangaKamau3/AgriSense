import ee
import time
from src.data.gee_fetcher import initialize_gee, get_temporal_composite
from src.processing.masking import mask_s2_clouds
from src.processing.indices import add_all_indices
from src.ml.model import CropStressModel
from src.ml.forecasting import CropHealthForecaster

def run_pipeline(roi_coords, use_ground_truth: bool = True, 
                ground_truth_path: str = 'data/sample_ground_truth.csv'):
    """
    Runs the full pipeline for a given ROI with pixel-wise stress classification.
    
    Args:
        roi_coords: Coordinates defining the region of interest
        use_ground_truth: Whether to use ground truth trained model (vs synthetic)
        ground_truth_path: Path to ground truth CSV file
    
    Returns:
        Dictionary with GEE objects for visualization and field statistics
    """
    print("Initializing Pipeline...")
    initialize_gee()
    
    roi = ee.Geometry.Polygon(roi_coords)
    
    print("Fetching temporal composite (Robust to clouds)...")
    # Use temporal composite instead of single image
    image = get_temporal_composite(roi)
    
    print("Processing image (Cloud Masking & Index Calculation)...")
    try:
        # Check if image has bands (by trying to compute indices)
        image = add_all_indices(image)
        
        print("Running Pixel-Wise Stress Detection Model...")
        # Initialize model (spatial features disabled by default for performance)
        model = CropStressModel(use_spatial_features=False)
        
        # Train model
        if use_ground_truth:
            try:
                print(f"Training model with ground truth data from {ground_truth_path}...")
                model.train_from_ground_truth(ground_truth_path, num_trees=50)
            except Exception as e:
                print(f"Warning: Could not load ground truth data: {e}")
                print("Falling back to synthetic training data...")
                model.train_synthetic(num_trees=30)
        else:
            print("Using synthetic training data...")
            model.train_synthetic(num_trees=30)
        
        # Classify image (pixel-wise)
        classified_image = model.classify(image)
        
        # Get field statistics
        print("Calculating field-level stress statistics...")
        try:
            field_stats = model.get_field_statistics(classified_image, roi)
            print("\nField Statistics:")
            print(f"Total Area: {field_stats['total_area_hectares']:.2f} hectares")
            for stress_type, stats in field_stats.items():
                if isinstance(stats, dict) and 'percentage' in stats:
                    print(f"  {stress_type}: {stats['percentage']:.1f}% ({stats['area_m2']:.0f} m²)")
        except Exception as e:
            print(f"Warning: Could not calculate field statistics: {e}")
            field_stats = None
        
        # Combine original image with classification
        result_image = image.addBands(classified_image)
        
        # --- FORECASTING STEP ---
        print("Running Crop Health Forecast (14-day prediction)...")
        chart_data = None
        try:
            forecaster = CropHealthForecaster()
            print("Forecaster initialized. Predicting future NDVI...")
            trend, predicted_ndvi = forecaster.predict_future_ndvi(roi)
            print("Prediction object created.")
            
            # Add forecast bands to result
            result_image = result_image.addBands(trend).addBands(predicted_ndvi)
            print("Forecast bands added successfully.")
            
            # Fetch Chart Data (Historical)
            print("Fetching trend chart data...")
            chart_data = forecaster.get_trend_chart_data(roi)
            print("Chart data fetched.")
        except Exception as e:
            print(f"Forecasting failed: {e}")
            import traceback
            traceback.print_exc()
            # We continue without forecast bands if this fails
        
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "roi": roi,
            "image": result_image,
            "chart_data": chart_data,
            "field_statistics": field_stats,
            "model_info": {
                "type": "pixel_wise_random_forest",
                "training_source": model.training_data_source,
                "num_trees": model.num_trees,
                "stress_classes": model.stress_classes
            }
        }
        
        print("Pipeline execution complete. Returning GEE objects.")
        return result
        
    except Exception as e:
        if "No band named" in str(e):
            raise Exception("No satellite data found for this region/time. \nHints:\n1. Check if coordinates are in [Longitude, Latitude] format.\n2. Ensure the region is on land.\n3. The region might be too cloudy recently.")
        raise e
