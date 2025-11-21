from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.pipeline.runner import run_pipeline

app = FastAPI(title="AgriSense API")

class ROIRequest(BaseModel):
    coordinates: List[List[float]]

@app.get("/")
def read_root():
    return {"message": "Welcome to AgriSense API"}

@app.post("/analyze")
def analyze_field(roi: ROIRequest):
    """
    Triggers the analysis pipeline for a given ROI.
    Returns a Tile URL for visualization.
    """
    try:
        result = run_pipeline(roi.coordinates)
        if result is None:
            raise HTTPException(status_code=404, detail="No data found for the given ROI")
        
        # Visualization parameters for the stress class band
        vis_params = {
            'min': 0,
            'max': 4,
            'palette': ['green', 'blue', 'red', 'yellow', 'brown'],
            'bands': ['stress_class']
        }
        
        # Get the MapID to generate a tile URL
        # We select the stress_class band for visualization
        map_id_dict = result['image'].select('stress_class').getMapId(vis_params)
        
        return {
            "timestamp": result['timestamp'],
            "tile_url": map_id_dict['tile_fetcher'].url_format,
            "attribution": "Map tiles by Google Earth Engine"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
