from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union
import sys
import os

# Try to import TestClient for testing
try:
    from fastapi.testclient import TestClient
except ImportError:
    # If TestClient is not available, create a dummy class
    class TestClient:
        def __init__(self, app):
            self.app = app
        def get(self, url):
            return type('Response', (), {'status_code': 200})()

# Add the parent directory of the current script to sys.path.
# This is crucial for allowing FastAPI to import other modules
# (like 'prediction') from the same 'ml_model' directory
# when you run the API from a different location (e.g., the project root).
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the prediction function from the prediction module
try:
    from prediction import predict_next_day_demand
except ImportError as e:
    print(f"Warning: Could not import prediction module: {e}")

# Initialize the FastAPI application
app = FastAPI(
    title="Fast-Commerce ML Prediction API",
    description="API for predicting next-day demand for fast-commerce inventory items "
                "based on historical sales data using Prophet or ARIMA models.",
    version="1.0.0"
)

# Pydantic models for request and response data validation and serialization.
# These define the expected structure of data sent to and received from the API.

class PredictionRequest(BaseModel):
    """
    Defines the structure of the request body for the /predict_demand endpoint.
    """
    pincode: str
    item: str
    model_type: str = "prophet" # Default model type, can be 'arima'

class PredictionResponse(BaseModel):
    """
    Defines the structure of the response body for the /predict_demand endpoint.
    """
    pincode: str
    item: str
    predicted_demand: int
    restock_quantity: int
    status: str
    message: Union[str, None] = None # Optional message for success or error details

@app.get("/")
async def root():
    """
    Root endpoint for the API. Returns a welcome message.
    """
    return {"message": "Welcome to the Fast-Commerce ML Prediction API! "
                       "Visit /docs for interactive API documentation."}

@app.get("/health")
async def health_check():
    """
    Health check endpoint for the API.
    """
    return {"status": "healthy", "service": "Fast-Commerce ML Prediction API"}

@app.post("/predict_demand", response_model=PredictionResponse)
async def get_predicted_demand(request: PredictionRequest):
    """
    Endpoint to predict next-day demand for a specific item in a given pincode.
    It uses the trained ML model (Prophet or ARIMA) to generate the forecast
    and suggests a restock quantity.

    Args:
        request (PredictionRequest): The request body containing pincode, item, and model_type.

    Returns:
        PredictionResponse: The predicted demand, suggested restock quantity, and status.

    Raises:
        HTTPException: If the prediction fails or the model is not found,
                       an appropriate HTTP error is returned.
    """
    try:
        print(f"Received prediction request for Pincode: {request.pincode}, "
              f"Item: {request.item}, Model Type: {request.model_type}")

        # Validate request parameters
        if not request.pincode or not request.item:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: pincode and item are required."
            )

        if request.model_type not in ['prophet', 'arima']:
            raise HTTPException(
                status_code=400,
                detail="Invalid model_type. Must be 'prophet' or 'arima'."
            )

        # Call the predict_next_day_demand function from the prediction module
        prediction_result = predict_next_day_demand(
            pincode=request.pincode,
            item=request.item,
            model_type=request.model_type
        )

        # Check the status of the prediction result
        if prediction_result["status"] != "Success":
            # If prediction failed, raise an HTTPException with appropriate status code
            raise HTTPException(
                status_code=404, # 404 Not Found is suitable if model isn't found or data issue
                detail={
                    "pincode": prediction_result.get("pincode"),
                    "item": prediction_result.get("item"),
                    "model_type": request.model_type,
                    "status": prediction_result.get("status"),
                    "message": prediction_result.get("status", "Prediction failed due to an unknown error.")
                }
            )

        # If prediction was successful, return the PredictionResponse
        return PredictionResponse(
            pincode=prediction_result["pincode"],
            item=prediction_result["item"],
            predicted_demand=prediction_result["predicted_demand"],
            restock_quantity=prediction_result["restock_quantity"],
            status=prediction_result["status"],
            message="Demand prediction successful."
        )
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        # Handle any other unexpected errors
        print(f"Unexpected error in prediction endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/models/available")
async def get_available_models():
    """
    Endpoint to get information about available trained models.
    """
    try:
        import os
        import glob
        
        model_dir = "trained_models"
        if not os.path.exists(model_dir):
            return {"available_models": [], "message": "No trained models found."}
        
        # Get all model files
        model_files = glob.glob(os.path.join(model_dir, "*.joblib"))
        
        models_info = []
        for model_file in model_files:
            filename = os.path.basename(model_file)
            # Parse filename to extract model info
            # Expected format: {model_type}_model_{pincode}_{item}.joblib
            parts = filename.replace('.joblib', '').split('_')
            if len(parts) >= 4:
                model_type = parts[0]
                pincode = parts[2]
                item = '_'.join(parts[3:])  # Handle items with underscores
                models_info.append({
                    "model_type": model_type,
                    "pincode": pincode,
                    "item": item,
                    "filename": filename
                })
        
        return {
            "available_models": models_info,
            "total_models": len(models_info)
        }
    except Exception as e:
        print(f"Error getting available models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model information: {str(e)}"
        )

# How to run this API:
# 1. Make sure you have uvicorn installed: `pip install uvicorn`
# 2. Navigate to the `ml_model` directory in your terminal (or the directory containing api_integration.py).
# 3. Run the following command:
#    `uvicorn api_integration:app --reload`
#    --reload enables auto-reloading of the server on code changes (useful for development).
# 4. Once running, open your web browser and go to `http://127.0.0.1:8000/docs`
#    This will open the interactive API documentation (Swagger UI), where you can
#    test the `/predict_demand` endpoint directly.
#
# Before running the API, ensure you have:
# - Run `model_training.py` at least once to train and save the models in the `trained_models` directory.
# - All required Python packages installed (from `requirements.txt`).

# Example of how to test this API programmatically (e.g., in a separate test script):
# import requests
#
# if __name__ == "__main__":
#     # This block will not run when uvicorn starts the app.
#     # It's for demonstrating how a client would interact with the API.
#     print("--- Example API Client Interaction ---")
#     api_url = "http://127.0.0.1:8000/predict_demand"
#
#     # Test Case 1: Successful prediction
#     payload_success = {
#         "pincode": "110037",
#         "item": "Milk",
#         "model_type": "prophet"
#     }
#     print(f"\nSending request: {payload_success}")
#     try:
#         response = requests.post(api_url, json=payload_success)
#         response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
#         print("Response (Success):", response.json())
#     except requests.exceptions.RequestException as e:
#         print(f"Request failed: {e}")
#         if hasattr(e, 'response') and e.response is not None:
#             print("Error Response:", e.response.json())
#
#     # Test Case 2: Prediction for a non-existent model
#     payload_fail = {
#         "pincode": "999999",
#         "item": "NonExistentItem",
#         "model_type": "arima"
#     }
#     print(f"\nSending request: {payload_fail}")
#     try:
#         response = requests.post(api_url, json=payload_fail)
#         response.raise_for_status()
#         print("Response (Failure - unexpected success):", response.json())
#     except requests.exceptions.RequestException as e:
#         print(f"Request failed (Expected failure): {e}")
#         if hasattr(e, 'response') and e.response is not None:
#             print("Error Response:", e.response.json())
#
#     print("\nAPI client interaction example complete.")
