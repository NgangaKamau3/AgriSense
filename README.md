# AgriSense 🛰️🌾

**Real-Time, Country-Scale Crop Stress Monitoring using Satellite Earth Observation & Machine Learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google-earth&logoColor=white)](https://earthengine.google.com/)

---

## 📖 Overview

**AgriSense** is an advanced agricultural intelligence system designed to detect early signs of crop stress (water, heat, nutrient, disease) using Sentinel-2 satellite imagery.

Unlike traditional tools that rely on simple thresholding, AgriSense Pro leverages **Google Earth Engine's (GEE) server-side Machine Learning** to perform robust, pixel-level classification at a country-wide scale. It uses temporal compositing to "see through" clouds and noise, providing reliable insights for farmers and agronomists.

### Key Features
- **🌍 Country-Scale Analysis**: Process petabytes of data on Google's infrastructure, not your laptop.
- **☁️ Cloud-Robust**: Automatic temporal median compositing removes clouds and shadows.
- **🧠 Server-Side ML**: Random Forest classifiers run directly on GEE for high-performance inference.
- **📊 Interactive Dashboard**: A Streamlit-based UI to visualize stress maps and trends instantly.

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
git clone https://github.com/yourusername/AgriSense.git
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

## 📂 Project Structure

```
AgriSense/
├── docs/               # Documentation (Tutorials, Roadmaps)
├── src/
│   ├── api/            # FastAPI Backend (Future expansion)
│   ├── dashboard/      # Streamlit Application
│   ├── data/           # GEE Data Fetching & Compositing
│   ├── ml/             # GEE-Native Machine Learning Models
│   ├── pipeline/       # Core Orchestration Logic
│   └── processing/     # Spectral Indices & Masking
├── run.py              # Unified Entry Point
├── requirements.txt    # Python Dependencies
└── README.md           # You are here
```

## 🤝 Contributing

We welcome contributions! Please read the [Production Roadmap](docs/roadmap.md) to see where we need help, particularly in:
- Integrating Ground Truth datasets.
- Expanding the spectral index library.
- Improving the frontend visualization.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
