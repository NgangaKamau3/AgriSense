import ee
import time
from src.data.gee_fetcher import initialize_gee, get_temporal_composite
from src.processing.masking import mask_s2_clouds
from src.processing.indices import add_all_indices
from src.ml.model import CropStressModel
from src.ml.forecasting import CropHealthForecaster

def run_pipeline(roi_coords):
    """
    Runs the full pipeline for a given ROI.
    Returns GEE objects for visualization.
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
        
        print("Running GEE-native Stress Detection Model...")
        model = CropStressModel()
        classified_image = model.classify(image)
        
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
            "chart_data": chart_data
        }
        
        print("Pipeline execution complete. Returning GEE objects.")
        return result
        
    except Exception as e:
        if "No band named" in str(e):
            raise Exception("No satellite data found for this region/time. \nHints:\n1. Check if coordinates are in [Longitude, Latitude] format.\n2. Ensure the region is on land.\n3. The region might be too cloudy recently.")
        raise e
