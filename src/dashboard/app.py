import streamlit as st
import ee
import folium
import streamlit.components.v1 as components
from src.pipeline.runner import run_pipeline
from src.data.gee_fetcher import initialize_gee

# Initialize GEE
try:
    initialize_gee()
except:
    st.error("Google Earth Engine authentication failed. Please authenticate locally.")

st.set_page_config(page_title="AgriSense Pro", layout="wide", page_icon="🌾")

# --- Sidebar Configuration ---
st.sidebar.title("🌾 AgriSense Pro")
st.sidebar.markdown("Real-time crop health monitoring from space.")

# Demo Locations (Kenya & Africa Focused)
DEMO_LOCATIONS = {
    "Custom Coordinates": None,
    "Mwea Irrigation Scheme 🇰🇪 (Rice)": [[37.34, -0.70], [37.36, -0.70], [37.36, -0.72], [37.34, -0.72], [37.34, -0.70]],
    "Kericho Tea Plantations 🇰🇪 (Tea)": [[35.28, -0.36], [35.30, -0.36], [35.30, -0.38], [35.28, -0.38], [35.28, -0.36]],
    "Uasin Gishu Maize Farms 🇰🇪 (Corn)": [[35.25, 0.50], [35.30, 0.50], [35.30, 0.45], [35.25, 0.45], [35.25, 0.50]],
    "Del Monte Thika 🇰🇪 (Pineapples)": [[37.05, -1.00], [37.10, -1.00], [37.10, -1.05], [37.05, -1.05], [37.05, -1.00]],
    "Naivasha Flower Farms 🇰🇪 (Floriculture)": [[36.40, -0.75], [36.45, -0.75], [36.45, -0.80], [36.40, -0.80], [36.40, -0.75]],
}

selected_loc = st.sidebar.selectbox("📍 Select a Location", list(DEMO_LOCATIONS.keys()), index=1)

if selected_loc == "Custom Coordinates":
    roi_input = st.sidebar.text_area(
        "Enter Polygon Coordinates",
        "[[-120.1, 36.1], [-120.1, 36.2], [-120.0, 36.2], [-120.0, 36.1], [-120.1, 36.1]]"
    )
else:
    roi_input = str(DEMO_LOCATIONS[selected_loc])
    st.sidebar.info(f"Loaded coordinates for {selected_loc}")

swap_coords = st.sidebar.checkbox("Swap Coordinates (Lat, Lon ➡️ Lon, Lat)", value=False, help="Check this if you copied coordinates from Google Maps (which uses Lat, Lon). GEE requires Lon, Lat.")

analyze_btn = st.sidebar.button("🚀 Analyze Region", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 How to read the map")
st.sidebar.markdown(
    """
    - 🟢 **Green**: Healthy Vegetation
    - 🔵 **Blue**: Water Stress (Thirsty)
    - 🔴 **Red**: Heat Stress (Overheating)
    - 🟡 **Yellow**: Nutrient Deficiency
    - 🟤 **Brown**: Disease / Bare Soil
    """
)

# --- Helper to add GEE Layer to Folium ---
def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

# Add the method to folium.Map
folium.Map.add_ee_layer = add_ee_layer

# --- Main Content ---
st.title("AgriSense Pro: Crop Intelligence 🛰️")
st.markdown(
    """
    **Welcome!** This tool uses satellite data to "see" invisible stress signals in crops. 
    It analyzes light reflection to determine if plants are healthy, thirsty, or sick.
    """
)

if analyze_btn:
    try:
        roi_coords = eval(roi_input)
        
        # Swap coordinates if requested
        if swap_coords:
            roi_coords = [[coord[1], coord[0]] for coord in roi_coords]
            st.sidebar.success(f"Swapped Coordinates: {roi_coords[0]}...")
        
        with st.spinner("🛰️ Contacting Google Earth Engine satellites... (This may take a moment)"):
            result = run_pipeline(roi_coords)
        
        if result:
            
            # --- TABS FOR DIFFERENT VIEWS ---
            tab1, tab2 = st.tabs(["🌱 Current Health", "🔮 Future Prediction"])
            
            with tab1:
                # --- Layout: Map on Left, Stats on Right ---
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🗺️ Current Field Stress Map")
                    
                    # Center map on the first point of the ROI
                    center_lat = roi_coords[0][1]
                    center_lon = roi_coords[0][0]
                    
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
                    
                    # Visualization Parameters
                    vis_params = {
                        'min': 0,
                        'max': 4,
                        'palette': ['green', 'blue', 'red', 'yellow', 'brown']
                    }
                    
                    # Add Stress Layer
                    m.add_ee_layer(result['image'].select('stress_class'), vis_params, 'Stress Classification')
                    
                    # Add ROI Polygon
                    folium.Polygon(
                        locations=[[coord[1], coord[0]] for coord in roi_coords], # Folium needs Lat, Lon
                        color='white',
                        weight=2,
                        fill=False
                    ).add_to(m)
                    
                    # Add Layer Control
                    folium.LayerControl().add_to(m)
                    
                    # Render directly to HTML string to avoid file locking
                    components.html(m._repr_html_(), height=600, scrolling=True)
                
                with col2:
                    st.subheader("📊 Field Health Report")
                    
                    # --- Real-time Analysis ---
                    with st.spinner("Calculating pixel statistics..."):
                        # Calculate pixel counts for each class
                        stats = result['image'].select('stress_class').reduceRegion(
                            reducer=ee.Reducer.frequencyHistogram(),
                            geometry=result['roi'],
                            scale=30,  # Optimized for speed
                            maxPixels=1e9,
                            bestEffort=True
                        ).getInfo()
                        
                        # Extract histogram
                        histogram = stats.get('stress_class', {})
                        total_pixels = sum(histogram.values())
                        
                        if total_pixels > 0:
                            # Calculate percentages
                            healthy_pct = (histogram.get('0', 0) / total_pixels) * 100
                            water_stress_pct = (histogram.get('1', 0) / total_pixels) * 100
                            heat_stress_pct = (histogram.get('2', 0) / total_pixels) * 100
                            nutrient_pct = (histogram.get('3', 0) / total_pixels) * 100
                            disease_pct = (histogram.get('4', 0) / total_pixels) * 100
                            
                            # Display Metrics
                            st.metric(label="Total Analyzed Area", value=f"{total_pixels * 100 / 10000:.2f} ha") # Approx hectares
                            
                            col_a, col_b = st.columns(2)
                            col_a.metric("Healthy Crop", f"{healthy_pct:.1f}%")
                            col_b.metric("Stressed Area", f"{100 - healthy_pct:.1f}%", delta_color="inverse")
                            
                            st.markdown("### 🧠 AI Diagnosis")
                            
                            # Dynamic Insights
                            if healthy_pct > 80:
                                st.success(f"✅ **Excellent Health**: {healthy_pct:.1f}% of the field is healthy.")
                            elif healthy_pct > 50:
                                st.warning(f"⚠️ **Moderate Stress**: {100 - healthy_pct:.1f}% of the field shows signs of stress.")
                            else:
                                st.error(f"🚨 **Critical Condition**: Majority of the field is stressed.")
                                
                            # Specific Issues
                            if water_stress_pct > 10:
                                st.info(f"💧 **Water Stress Detected**: {water_stress_pct:.1f}% of crops are thirsty.")
                            if heat_stress_pct > 5:
                                st.error(f"🔥 **Heat Stress Detected**: {heat_stress_pct:.1f}% of crops are overheating.")
                            if disease_pct > 5:
                                st.error(f"🦠 **Disease Risk**: {disease_pct:.1f}% of crops show irregular signatures.")

                        else:
                            st.warning("No pixels found in this region (possibly water or clouds).")
                    
                    with st.expander("ℹ️ What does this mean?"):
                        st.write("""
                        **Water Stress**: The plants are reflecting light that indicates low moisture content. Consider irrigation.
                        
                        **Heat Stress**: High surface temperature detected relative to the canopy.
                        
                        **Disease**: Irregular spectral signatures often associated with fungal or bacterial infection.
                        """)

            with tab2:
                st.subheader("🔮 14-Day Crop Health Forecast")
                st.markdown("Using **Linear Regression** on the last 60 days of satellite data to predict future trends.")
                
                if result.get('chart_data') is None:
                    st.warning("⚠️ Forecast unavailable. Not enough clear historical data found for this region to generate a reliable prediction.")
                else:
                    col_pred_1, col_pred_2 = st.columns([2, 1])
                    
                    with col_pred_1:
                        # Forecast Map
                        center_lat = roi_coords[0][1]
                        center_lon = roi_coords[0][0]
                        m_pred = folium.Map(location=[center_lat, center_lon], zoom_start=13)
                        
                        # Predicted NDVI Vis Params
                        ndvi_vis = {
                            'min': 0.2,
                            'max': 0.8,
                            'palette': ['red', 'yellow', 'green']
                        }
                        
                        m_pred.add_ee_layer(result['image'].select('Predicted_NDVI'), ndvi_vis, 'Predicted Health (NDVI)')
                        folium.LayerControl().add_to(m_pred)
                        components.html(m_pred._repr_html_(), height=500, scrolling=True)
                        
                    with col_pred_2:
                        st.markdown("### 📈 Health Trend")
                        
                        # Process Chart Data
                        import pandas as pd
                        features = result['chart_data']['features']
                        if features:
                            data = [{'Date': f['properties']['date'], 'NDVI': f['properties']['ndvi']} for f in features]
                            df = pd.DataFrame(data)
                            df['Date'] = pd.to_datetime(df['Date'])
                            df = df.sort_values('Date')
                            
                            st.line_chart(df.set_index('Date'))
                            
                            # Trend Analysis
                            latest_ndvi = df.iloc[-1]['NDVI']
                            start_ndvi = df.iloc[0]['NDVI']
                            trend_delta = latest_ndvi - start_ndvi
                            
                            if trend_delta > 0.05:
                                st.success("📈 **Improving Trend**: Crop health is improving over time.")
                            elif trend_delta < -0.05:
                                st.error("📉 **Declining Trend**: Crop health is deteriorating. Action needed!")
                            else:
                                st.info("➡️ **Stable Trend**: Crop health is stable.")
                        else:
                            st.warning("Not enough clear data points for a trend chart.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please check your coordinates or GEE authentication.")

else:
    st.info("👈 Select a location in the sidebar and click **Analyze Region** to start.")
