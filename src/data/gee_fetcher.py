import ee
import datetime

def initialize_gee():
    """Initializes the Earth Engine API."""
    try:
        ee.Initialize()
        print("Google Earth Engine initialized successfully.")
    except ee.EEException as e:
        if "project" in str(e).lower():
            print("\n[ERROR] GEE Project not set.")
            print("Please run: `earthengine set_project <YOUR_PROJECT_ID>` in your terminal.")
            print("If you don't have a project, create one at https://console.cloud.google.com/\n")
        raise e
    except Exception as e:
        print(f"Failed to initialize Google Earth Engine: {e}")
        print("Try running `earthengine authenticate` in your terminal.")
        raise e

def get_sentinel2_collection(roi, start_date, end_date, cloud_percentage=20):
    """
    Fetches Sentinel-2 Surface Reflectance collection filtered by ROI, date, and cloud cover.
    """
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_percentage))
    
    return s2

def get_temporal_composite(roi, days=45):
    """
    Fetches a temporal median composite for the ROI.
    This is robust against clouds and noise.
    """
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Relaxed cloud filter to 80% because median composite handles clouds well
    collection = get_sentinel2_collection(roi, start_date, end_date, cloud_percentage=80)
    
    # Median composite: For each pixel, take the median value across all images in the stack.
    # This effectively removes clouds (which are bright outliers) and shadows (dark outliers).
    composite = collection.median().clip(roi)
    
    # Add a time property for metadata
    return composite.set('system:time_start', ee.Date(end_date).millis())
