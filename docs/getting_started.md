# AgriSense Pro Walkthrough

## Overview
AgriSense Pro is a production-grade crop stress monitoring system. It leverages Google Earth Engine's server-side machine learning to perform country-scale analysis with robust handling of messy data.

## Key Features
- **Temporal Compositing**: Automatically removes clouds and noise by taking the median of images over 15 days.
- **Server-Side ML**: Runs Random Forest classification on Google's servers, enabling analysis of massive areas without local memory limits.
- **Interactive Map**: Visualizes stress classes directly on the map with a legend.

## Steps to Run

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Authenticate Google Earth Engine**
    ```bash
    earthengine authenticate
    ```

3.  **Run the Dashboard**
    ```bash
    python run.py --mode dashboard
    ```

4.  **Using the Dashboard**
    - Enter coordinates for any region (even a whole city!).
    - Click "Analyze Region".
    - The map will display the stress classification layer.
    - **Legend**:
        - 🟢 **Green**: Healthy
        - 🔵 **Blue**: Water Stress
        - 🔴 **Red**: Heat Stress
        - 🟡 **Yellow**: Nutrient Deficiency
        - 🟤 **Brown**: Disease

## Troubleshooting
- **GEE Error**: Ensure you are authenticated.
- **Slow Map**: Large areas may take a few seconds to render as GEE computes the tiles on the fly.
