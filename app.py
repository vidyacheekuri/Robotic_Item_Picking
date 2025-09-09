from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import io
import cv2

# Import the prediction function
from predict import predict_and_visualize

# Create a FastAPI app instance
app = FastAPI(title="6D Pose Estimation API")

# This line tells FastAPI to serve all files from the 'frontend' directory
# when a request comes in for the root path "/".
#app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# --- Create an endpoint to serve the index.html file ---
@app.get("/")
async def read_index():
    """Serves the main HTML page of the web app."""
    return FileResponse('frontend/index.html')


@app.post("/predict/", response_class=StreamingResponse)
def create_upload_file(file: UploadFile = File(...)):
    """
    Accepts an image, runs pose prediction, draws the result,
    and returns the final image.
    """
    temp_image_path = f"temp_{file.filename}"
    
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get the image with the visualization drawn on it
    result_image_np = predict_and_visualize(temp_image_path)
    
    os.remove(temp_image_path)
    
    # Encode the resulting numpy array image to a byte stream
    is_success, buffer = cv2.imencode(".jpg", result_image_np)
    
    # Return the byte stream as a streaming response
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
