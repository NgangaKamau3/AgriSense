# Building AgriSense: A Deep Dive into Real-Time Crop Stress Detection from Space 🛰️🌾

**Welcome, Earth Defenders and Data Wizards!**

Here, we are going to embark on a journey to build **AgriSense**, a system that monitors the health of crops from space. We'll combine the massive scale of **Google Earth Engine (GEE)** with the precision of **Machine Learning** to detect early signs of stress in plants—whether they are thirsty, overheating, or fighting off disease.

By the end of this tutorial, you will understand how we built a pipeline that takes raw satellite pixels and turns them into actionable insights for farmers, all running on a standard laptop.

---

## 1. The Mission: Why Crop Stress? 🎯

Plants "talk" to us through light. Healthy vegetation reflects light differently than stressed vegetation.
- **Healthy plants** absorb red light (for photosynthesis) and strongly reflect Near-Infrared (NIR) light.
- **Stressed plants** often reflect less NIR and more visible red light.

Our goal is to capture these signals using **Sentinel-2 satellites**, calculate mathematical indices (like NDVI), and feed them into an AI model to classify the stress type.

---

## 2. The Tech Stack 🛠️

We chose a stack that balances power with accessibility:

*   **Google Earth Engine (GEE)**: The engine room. It processes petabytes of satellite data in the cloud, so your laptop doesn't have to.
*   **Python**: The glue. We use it to orchestrate GEE and run our local logic.
*   **Sentinel-2**: Our eyes in the sky. These satellites revisit the same spot every 5 days with 10-20m resolution.
*   **Scikit-Learn**: For our Machine Learning brain (Random Forest).
*   **Streamlit**: For building a beautiful, interactive dashboard in minutes.

---

## 3. Step-by-Step Implementation 🚀

### Phase 1: Data Acquisition (The "Eyes")

We need to fetch clear images of our field. Clouds are the enemy of optical remote sensing.

**The Strategy:**
1.  Define a **Region of Interest (ROI)** (the farm).
2.  Filter the Sentinel-2 collection by date and cloud cover.
3.  Pick the most recent image.

*Code Highlight (`src/data/gee_fetcher.py`):*
```python
def get_sentinel2_collection(roi, start_date, end_date):
    return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
```

### Phase 2: Preprocessing (The "Lens")

Raw data isn't enough. We need to clean it and extract features.

**Cloud Masking:**
Even with filtering, some clouds sneak in. We use the **QA60 band** (Quality Assurance) to identify and mask out cloudy pixels so they don't confuse our model.

**Spectral Indices:**
This is where the magic happens. We calculate five key indices:
1.  **NDVI (Normalized Difference Vegetation Index)**: General plant health.
2.  **NDWI (Normalized Difference Water Index)**: Water content in leaves.
3.  **NDMI (Normalized Difference Moisture Index)**: Deep moisture stress.
4.  **MSI (Moisture Stress Index)**: Another angle on water stress.
5.  **NDRE (Normalized Difference Red Edge)**: Chlorophyll content (great for detecting nutrient issues).

*Code Highlight (`src/processing/indices.py`):*
```python
def calculate_ndvi(image):
    # NDVI = (NIR - Red) / (NIR + Red)
    return image.normalizedDifference(['B8', 'B4']).rename('NDVI')
```

### Phase 3: The Pipeline & Optimization (The "Brain") 🧠

**The Challenge:**
Satellite images are huge. Downloading them to process locally would crash a standard laptop and take forever.

**The Solution:**
We perform all the heavy lifting **on Google's servers**. We calculate the indices and mask clouds *in the cloud*. Then, we only ask GEE for the **average statistics** of the field (e.g., "Mean NDVI is 0.75").

This reduces gigabytes of image data to a tiny dictionary of numbers:
`{'NDVI': 0.75, 'NDWI': 0.4, ...}`

This is why AgriSense is so fast and lightweight!

*Code Highlight (`src/pipeline/runner.py`):*
```python
stats = image.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=roi,
    scale=10,
    maxPixels=1e8,
    bestEffort=True # Optimization magic!
)
```

### Phase 4: Machine Learning (The "Doctor") 🩺

We use a **Random Forest Classifier**. Why? Because it handles non-linear relationships between our indices well and is robust against overfitting.

**Training:**
We train the model on these aggregated field statistics.
- **Input**: [NDVI, NDWI, NDMI, MSI, NDRE]
- **Output**: Stress Class (Healthy, Water Stress, Heat Stress, etc.)

*Code Highlight (`src/ml/model.py`):*
```python
self.model = RandomForestClassifier(n_estimators=10)
self.model.fit(X, y)
```

### Phase 5: The Dashboard (The "Face") 🖥️

Finally, we wrap it all up in **Streamlit**. It allows us to create a web app with just Python. Users can input coordinates, click "Analyze", and see the results instantly on an interactive map.

---

## 4. Running AgriSense Yourself 🏃‍♂️

Want to try it?

1.  **Clone the repo** (or use the files provided).
2.  **Install dependencies**: `pip install -r requirements.txt`
3.  **Authenticate GEE**: `earthengine authenticate`
4.  **Run it**: `python run.py --mode dashboard`

---

## Conclusion

AgriSense demonstrates the power of modern geospatial analysis. By leveraging cloud computing for processing and local machine learning for inference, we created a scalable, real-time monitoring tool that is accessible to anyone with a laptop.

**Happy Farming!** 🌱
