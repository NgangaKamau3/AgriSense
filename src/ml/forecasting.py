import ee
import datetime
from src.data.gee_fetcher import get_sentinel2_collection
from src.processing.masking import mask_s2_clouds

class CropHealthForecaster:
    def __init__(self):
        pass

    def predict_future_ndvi(self, roi, days_history=60, days_future=14):
        """
        Predicts NDVI 14 days into the future using Linear Regression on the last 60 days.
        Returns:
            - trend_image: Slope of NDVI (Rate of change).
            - predicted_image: Estimated NDVI at T + days_future.
        """
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days_history)
        
        # Fetch collection
        collection = get_sentinel2_collection(
            roi, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d'), 
            cloud_percentage=80
        )
        
        # Preprocess: Cloud Mask + NDVI + Time Band
        def preprocess(image):
            image = mask_s2_clouds(image)
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            
            # Add time band (x-axis for regression)
            # Scale time to avoid huge numbers (e.g., days since start)
            time_band = image.metadata('system:time_start').divide(1000 * 60 * 60 * 24).rename('time')
            
            return image.addBands(ndvi).addBands(time_band)

        processed_col = collection.map(preprocess).select(['time', 'NDVI'])
        
        # Run Linear Regression (Least Squares)
        # Output bands: 'scale' (slope), 'offset' (intercept)
        linear_fit = processed_col.reduce(ee.Reducer.linearFit())
        
        # Calculate Future Time (x_future)
        # We need 'days since epoch' to match the training data
        epoch = datetime.date(1970, 1, 1)
        days_since_epoch = (end_date - epoch).days + days_future
        future_time = ee.Number(days_since_epoch)
        
        # Predict NDVI = offset + scale * future_time
        predicted_ndvi = linear_fit.select('offset').add(
            linear_fit.select('scale').multiply(future_time)
        ).rename('Predicted_NDVI')
        
        return linear_fit.select('scale').rename('NDVI_Trend'), predicted_ndvi

    def get_trend_chart_data(self, roi, days_history=60):
        """
        Fetches historical NDVI data for charting.
        """
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days_history)
        
        collection = get_sentinel2_collection(
            roi, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d'), 
            cloud_percentage=60
        )
        
        def get_stats(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            mean_ndvi = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=30, # Optimized for speed
                maxPixels=1e9,
                bestEffort=True
            ).get('NDVI')
            return ee.Feature(None, {
                'date': image.date().format('yyyy-MM-dd'),
                'ndvi': mean_ndvi
            })
            
        # Filter out images where mean_ndvi is null (too cloudy)
        timeseries = collection.map(mask_s2_clouds).map(get_stats).filter(ee.Filter.notNull(['ndvi']))
        
        return timeseries.getInfo()
