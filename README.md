# AgriSense 🛰️🌾

**Real-Time, Country-Scale Crop Stress Monitoring using Satellite Earth Observation & Machine Learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google-earth&logoColor=white)](https://earthengine.google.com/)

---

## 📖 Overview

**AgriSense** is an advanced agricultural intelligence system designed to detect early signs of crop stress (water, heat, nutrient, disease) using Sentinel-2 satellite imagery.

Unlike traditional tools that rely on simple thresholding, AgriSense leverages **Google Earth Engine's (GEE) server-side Machine Learning** with **pixel-wise classification** trained on real ground truth data. It uses temporal compositing to "see through" clouds and noise, providing reliable, spatially-aware insights for farmers and agronomists.

### Key Features
- **🎯 Pixel-Level Precision**: 10m resolution stress detection - know exactly where in your field the problem is
- **📊 Ground Truth Trained**: Models learn from real labeled data, not hardcoded rules
- **🗺️ Spatial Awareness**: Detects stress hotspots and boundaries using neighboring pixel context
- **🌍 Country-Scale Analysis**: Process petabytes of data on Google's infrastructure
- **☁️ Cloud-Robust**: Automatic temporal median compositing removes clouds and shadows
- **🧠 Advanced ML**: Random Forest with spatial features and comprehensive validation metrics
- **📈 Field Statistics**: Get actionable insights - "27% of your field shows water stress"
- **📊 Interactive Dashboard**: Streamlit-based UI to visualize stress maps and trends instantly


---

## 📚 Documentation

We believe in comprehensive documentation. Whether you are a beginner or a senior engineer, we have a guide for you:

1.  **[Getting Started](docs/getting_started.md)** 🏁
    *   Installation, Authentication, and Running the Dashboard.
    *   *Start here if you just want to run the app.*

2.  **[Tutorial: Building AgriSense](docs/tutorial.md)** 🎓
    *   A deep dive into the "Why" and "How".
    *   Explains the physics of remote sensing, spectral indices (NDVI, NDWI), and the architecture choices.
    *   *Read this to understand the science.*

3.  **[Production Roadmap & Architecture](docs/roadmap.md)** 🏗️
    *   Gap analysis between prototype and production.
    *   Strategies for integrating Ground Truth data.
    *   *Read this if you are an architect or contributor looking to scale the system.*

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/NgangaKamau3/AgriSense.git
cd AgriSense

# 2. Install dependencies
pip install -r requirements.txt

# 3. Authenticate Google Earth Engine
earthengine authenticate

# 4. Run the Dashboard
python run.py --mode dashboard

# 5. Run the API Backend (Optional)
python run.py --mode api
```

## 🤖 ML Model Architecture

AgriSense now uses a **pixel-wise Random Forest classifier** trained on real ground truth data:

### Training with Ground Truth Data

1. **Prepare your ground truth data** in CSV format:
   ```csv
   longitude,latitude,date,crop_type,NDVI,NDWI,NDMI,MSI,NDRE,stress_class,confidence,source
   36.8219,-1.2921,2024-01-15,wheat,0.82,0.35,0.48,0.52,0.68,0,high,field_survey
   ```

2. **Stress classes**:
   - `0`: Healthy
   - `1`: Water Stress
   - `2`: Heat Stress
   - `3`: Nutrient Deficiency
   - `4`: Disease

3. **Train the model**:
   ```python
   from src.ml.model import CropStressModel
   
   model = CropStressModel(use_spatial_features=False)
   model.train_from_ground_truth('data/your_ground_truth.csv', num_trees=50)
   metrics = model.validate()
   ```

4. **Quick test**:
   ```bash
   python test_model.py
   ```

### Model Features

- **Spatial Context**: Considers neighboring pixels (3x3 window) for better accuracy
- **Texture Features**: GLCM features detect patterns invisible to spectral indices alone
- **Comprehensive Metrics**: Precision, recall, F1-score per stress class
- **Field Statistics**: Automatic calculation of stress distribution (area and %)
- **Confidence Scores**: Per-pixel confidence for prediction reliability

See [walkthrough.md](.gemini/antigravity/brain/b3a0fd0b-50ed-49b2-9edb-8b3394369e96/walkthrough.md) for detailed implementation guide.

## 📂 Project Structure

```
AgriSense/
├── data/                  # Ground truth training data
│   └── sample_ground_truth.csv
├── docs/                  # Documentation (Tutorials, Roadmaps)
├── src/
│   ├── api/               # FastAPI Backend (Future expansion)
│   ├── dashboard/         # Streamlit Application
│   ├── data/              # GEE Data Fetching & Ground Truth Loading
│   ├── ml/                # ML Models & Feature Engineering
│   │   ├── model.py       # Pixel-wise Random Forest
│   │   ├── feature_engineering.py  # Spatial & temporal features
│   │   └── forecasting.py # NDVI trend prediction
│   ├── pipeline/          # Core Orchestration Logic
│   └── processing/        # Spectral Indices & Masking
├── test_model.py          # Quick model verification script
├── run.py                 # Unified Entry Point
├── requirements.txt       # Python Dependencies
└── README.md              # You are here
```

## 🤝 Contributing

We welcome contributions! Please read the [Production Roadmap](docs/roadmap.md) to see where we need help, particularly in:
- Integrating Ground Truth datasets.
- Expanding the spectral index library.
- Improving the frontend visualization.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
