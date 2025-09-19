import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime
import pytz
from transformers import pipeline
import torch
from pydantic import BaseModel
from typing import Union
import csv

# Initialize FastAPI app
app = FastAPI(title="Living Portfolio API", version="1.0")

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI pipelines (assuming they are loaded successfully)
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    text_generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Error loading models: {e}")
    sentiment_analyzer = None
    text_generator = None

# Define paths for denoising images and mount static files
os.makedirs("static/cleaned_images", exist_ok=True)
os.makedirs("static/uploaded_images", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Denoising function
def denoise_image(image):
    # Perform denoising
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=20, templateWindowSize=17, searchWindowSize=28)
    
    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Create an output image to hold the components
    output_image = np.zeros_like(binary_image)
    
    # Iterate through each component
    for i in range(1, num_labels):
        # Create a mask for the current component
        component_mask = (labels == i).astype("uint8") * 255
        # Keep components larger than a certain threshold size to preserve more details
        if stats[i, cv2.CC_STAT_AREA] > 50:
            output_image = cv2.bitwise_or(output_image, component_mask)
            
    # Invert the image back to original white objects
    output_image = cv2.bitwise_not(output_image)
    
    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(output_image, -1, kernel)
    
    # Adjust contrast and brightness
    alpha = 1.5  # Contrast control
    beta = 50    # Brightness control
    enhanced_image = cv2.convertScaleAbs(sharpened_image, alpha=alpha, beta=beta)
    
    # Automatic border detection and cropping
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crop based on the largest contour found
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_img = enhanced_image[y:y+h, x:x+w]
    else:
        cropped_img = enhanced_image  # In case no contours are found
        
    return cropped_img

# HTML for the UI (Denoising page)
denoise_html_content = """
<!DOCTYPE html>
<html>
    <head>
        <title>Denoising Musical Sheets</title>
        <style>
            body { font-family: 'Inter', sans-serif; padding: 2rem; background-color: #0f172a; color: #f1f5f9; line-height: 1.6; }
            .container { max-width: 800px; margin: 0 auto; text-align: center; }
            h1 { color: #6366f1; font-size: 2.5rem; margin-bottom: 0.5rem; }
            .subtitle { color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem; }
            .upload-box { background-color: #1e293b; padding: 3rem 1.5rem; border-radius: 10px; border: 2px dashed #6366f1; transition: all 0.3s ease; }
            .upload-box:hover { border-color: #8b5cf6; }
            .upload-text { font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; }
            .upload-options { display: flex; flex-direction: column; gap: 1rem; align-items: center; }
            input[type="file"] { display: none; }
            label.upload-btn, .sample-btn {
                padding: 0.75rem 2rem;
                background-image: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: bold;
                font-size: 1rem;
                transition: all 0.3s ease;
                display: inline-block;
            }
            label.upload-btn:hover, .sample-btn:hover { box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4); transform: translateY(-2px); }
            .upload-or { color: #94a3b8; margin: 1rem 0; font-size: 1rem; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Optical Music Recognition</h1>
            <p class="subtitle">An AI-powered Denoising Tool</p>
            <div class="upload-box">
                <p class="upload-text">Upload your image to get started</p>
                <div class="upload-options">
                    <form id="upload-form" action="/upload_and_denoise/" enctype="multipart/form-data" method="post">
                        <input name="file" type="file" id="file-input" accept="image/*" onchange="document.getElementById('upload-form').submit();">
                        <label for="file-input" class="upload-btn">Choose File</label>
                    </form>
                    <div class="upload-or">OR</div>
                    <form id="sample-form" action="/upload_and_denoise_sample/" method="post">
                        <button type="submit" class="sample-btn">Try with Sample Image</button>
                    </form>
                </div>
            </div>
        </div>
    </body>
</html>
"""

@app.get("/")
async def root():
    return {"message": "Living Portfolio API is running!"}

@app.get("/denoise-demo/", response_class=HTMLResponse)
async def read_denoise_root():
    return denoise_html_content

# Endpoint for handling regular file uploads
@app.post("/upload_and_denoise/", response_class=HTMLResponse)
async def upload_and_denoise(file: UploadFile = File(...)):
    # Read the file content
    file_content = await file.read()
    filename = file.filename
    
    # Save original image
    uploaded_image_path = os.path.join("static/uploaded_images", filename)
    with open(uploaded_image_path, "wb") as f:
        f.write(file_content)

    # Process the image
    image = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_GRAYSCALE)
    cleaned_image = denoise_image(image)

    # Save cleaned image
    cleaned_image_path = os.path.join("static/cleaned_images", 'cleaned_' + filename)
    cv2.imwrite(cleaned_image_path, cleaned_image)

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Denoised Image</title>
            <style>
                body {{ font-family: 'Inter', sans-serif; padding: 2rem; background-color: #0f172a; color: #f1f5f9; line-height: 1.6; }}
                .container {{ max-width: 900px; margin: 0 auto; text-align: center; }}
                h1 {{ color: #6366f1; font-size: 2.5rem; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin-top: 2rem; }}
                .image-box {{ background-color: #1e293b; padding: 1rem; border-radius: 10px; border: 1px solid #6366f1; }}
                .image-box h2 {{ margin-top: 0; font-size: 1.2rem; color: #f1f5f9; }}
                .image-box img {{ max-width: 100%; height: auto; border-radius: 5px; }}
                .back-link {{ display: inline-block; text-align: center; margin-top: 2rem; color: #8b5cf6; text-decoration: none; font-size: 1.1rem; padding: 0.75rem 2rem; border: 2px solid #8b5cf6; border-radius: 25px; transition: all 0.3s ease; }}
                .back-link:hover {{ background-color: #8b5cf6; color: #fff; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Denoising Complete!</h1>
                <div class="image-grid">
                    <div class="image-box">
                        <h2>Original Image</h2>
                        <img src="/static/uploaded_images/{filename}" alt="Original Musical Sheet">
                    </div>
                    <div class="image-box">
                        <h2>Denoised Image</h2>
                        <img src="/static/cleaned_images/cleaned_{filename}" alt="Cleaned Musical Sheet">
                    </div>
                </div>
                <a href="/denoise-demo/" class="back-link">Upload another image</a>
            </div>
        </body>
    </html>
    """)

# New endpoint for handling the sample image request
@app.post("/upload_and_denoise_sample/", response_class=HTMLResponse)
async def upload_and_denoise_sample():
    filename = "noised.jpg"
    filepath = os.path.join("static", "uploaded_images", filename) # Updated path
    
    # Check if the sample file exists
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Sample image not found in static/uploaded_images directory.")

    # Read the file content
    with open(filepath, "rb") as f:
        file_content = f.read()

    # Process the image
    image = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_GRAYSCALE)
    cleaned_image = denoise_image(image)

    # Save cleaned image
    cleaned_image_path = os.path.join("static/cleaned_images", 'cleaned_' + filename)
    cv2.imwrite(cleaned_image_path, cleaned_image)

    # Return HTML response with both original and cleaned images
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Denoised Image</title>
            <style>
                body {{ font-family: 'Inter', sans-serif; padding: 2rem; background-color: #0f172a; color: #f1f5f9; line-height: 1.6; }}
                .container {{ max-width: 900px; margin: 0 auto; text-align: center; }}
                h1 {{ color: #6366f1; font-size: 2.5rem; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin-top: 2rem; }}
                .image-box {{ background-color: #1e293b; padding: 1rem; border-radius: 10px; border: 1px solid #6366f1; }}
                .image-box h2 {{ margin-top: 0; font-size: 1.2rem; color: #f1f5f9; }}
                .image-box img {{ max-width: 100%; height: auto; border-radius: 5px; }}
                .back-link {{ display: inline-block; text-align: center; margin-top: 2rem; color: #8b5cf6; text-decoration: none; font-size: 1.1rem; padding: 0.75rem 2rem; border: 2px solid #8b5cf6; border-radius: 25px; transition: all 0.3s ease; }}
                .back-link:hover {{ background-color: #8b5cf6; color: #fff; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Denoising Complete!</h1>
                <div class="image-grid">
                    <div class="image-box">
                        <h2>Original Image</h2>
                        <img src="/static/uploaded_images/{filename}" alt="Original Musical Sheet">
                    </div>
                    <div class="image-box">
                        <h2>Denoised Image</h2>
                        <img src="/static/cleaned_images/cleaned_{filename}" alt="Cleaned Musical Sheet">
                    </div>
                </div>
                <a href="/denoise-demo/" class="back-link">Upload another image</a>
            </div>
        </body>
    </html>
    """)

# Rest of the original API endpoints
class LocationData(BaseModel):
    latitude: float
    longitude: float
# New Pydantic model for contact form data
class ContactForm(BaseModel):
    name: str
    email: str
    subject: str
    body: str

class Project(BaseModel):
    id: int
    title: str
    description: str
    tags: list
    github_url: str
    live_url: str = None

# Sample projects data (your original list)
projects_data = [
    {
        "id": 1,
        "title": "Optical Music Recognition (OMR) System",
        "description": "Developed an AI-based system to automatically recognize and digitize musical notation from sheet music.",
        "tags": ["Python", "TensorFlow", "CNN", "OpenCV"],
        "github_url": "https://github.com/Yashwanth1524/OMR"
    },
    {
        "id": 2,
        "title": "Denoising Musical Sheet",
        "description": "An API using FastAPI and OpenCV to clean and enhance noisy or damaged musical sheet images for improved recognition.",
        "tags": ["Python", "FastAPI", "OpenCV"],
        "github_url": "https://github.com/Yashwanth1524/OMR",
        "live_url": "http://127.0.0.1:8000/denoise-demo/"
    },
    {
        "id": 3,
        "title": "E-commerce Application with Servlets",
        "description": "A comprehensive e-commerce platform built on Java servlets for server-side logic and database interaction.",
        "tags": ["Java", "Servlets", "JSP", "HTML/CSS", "MySQL"],
        "github_url": "https://github.com/Yashwanth1524/E-commerce-Application-with-Servlets"
    },
    {
        "id": 4,
        "title": "Blog Website with ReactJS and Firebase",
        "description": "A dynamic blog website with user authentication and real-time content management, powered by React and Firebase.",
        "tags": ["React", "Firebase", "HTML/CSS"],
        "github_url": "https://github.com/Yashwanth1524/socio"
    }
]

@app.get("/projects")
async def get_projects():
    return projects_data

@app.post("/get-context/")
async def get_context(location: LocationData):
    try:
        # Get weather data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={location.latitude}&longitude={location.longitude}&current_weather=true"
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()
        current_weather = weather_data['current_weather']
        
        is_day = current_weather['is_day']
        temperature = current_weather['temperature']
        weather_code = current_weather['weathercode']

        # Determine context
        context = "default"
        if is_day == 0:
            context = "night"
        elif weather_code >= 51 and weather_code < 80:
            context = "rainy"
        elif weather_code >= 95:
            context = "stormy"
        elif temperature > 30:
            context = "hot-day"
        else:
            context = "sunny-day"

        # Get AI-generated message
        ai_message = generate_ai_message(context, temperature)

        return {
            "context": context,
            "theme": get_theme(context),
            "featured_project": get_featured_project(context),
            "ai_message": ai_message,
            "weather_data": {
                "is_day": bool(is_day),
                "temperature": temperature,
                "weather_code": weather_code
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing context: {str(e)}")

# UPDATED: Replaced email logic with file writing
@app.post("/send-email/")
async def send_email(contact_form: ContactForm):
    try:
        # Define the directory and CSV file path
        submissions_dir = "contact_submissions"
        
        # Check if the directory exists and create it if not
        if not os.path.exists(submissions_dir):
            os.makedirs(submissions_dir)
            print(f"Created directory: {submissions_dir}")

        csv_file_path = os.path.join(submissions_dir, "contact_submissions.csv")

        # Check if the file exists to decide whether to write headers
        file_exists = os.path.exists(csv_file_path)

        # Open the CSV file in append mode
        with open(csv_file_path, "a", newline="", encoding="utf-8") as file:
            fieldnames = ["timestamp", "name", "email", "subject", "message"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write the header only if the file is new
            if not file_exists:
                writer.writeheader()
            
            # Prepare the data row
            data_row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "name": contact_form.name,
                "email": contact_form.email,
                "subject": contact_form.subject,
                "message": contact_form.body
            }
            
            # Write the data to the CSV file
            writer.writerow(data_row)
            
        print(f"Contact form data appended to {csv_file_path}")
        return {"message": "Message saved successfully!"}

    except Exception as e:
        print(f"Error saving contact form data: {e}")
        # Return a 500 status code to the frontend to signal failure
        raise HTTPException(status_code=500, detail=f"Error saving message: {str(e)}")

@app.post("/analyze-sentiment/")
async def analyze_sentiment(text: str):
    if not sentiment_analyzer:
        return {"sentiment": "neutral", "confidence": 0.0}
    
    try:
        result = sentiment_analyzer(text[:512])[0]  # Limit input length
        return {
            "sentiment": result['label'].lower(),
            "confidence": float(result['score'])
        }
    except Exception as e:
        return {"sentiment": "neutral", "confidence": 0.0}

def generate_ai_message(context: str, temperature: float) -> str:
    if not text_generator:
        return f"Welcome! It's a {context} day. Perfect for exploring AI projects!"
    
    try:
        prompt = f"Create a creative welcome message for an AI engineer's portfolio website on a {context} day with temperature {temperature}°C:"
        result = text_generator(prompt, max_length=50, num_return_sequences=1, temperature=0.8)
        return result[0]['generated_text'].replace(prompt, "").strip()
    except Exception as e:
        return f"Exploring AI possibilities on this {context} day. Temperature: {temperature}°C"

def get_theme(context: str) -> dict:
    themes = {
        "default": {
            "--bg-color": "#1a1a2e", "--text-color": "#e6e6e6", 
            "--accent-color": "#4cc9f0", "--card-bg": "rgba(255,255,255,0.1)"
        },
        "night": {
            "--bg-color": "#0f0f1f", "--text-color": "#a0a0d0", 
            "--accent-color": "#7b68ee", "--card-bg": "rgba(160,160,208,0.15)"
        },
        "rainy": {
            "--bg-color": "#2b4162", "--text-color": "#f0f8ff", 
            "--accent-color": "#a0d2db", "--card-bg": "rgba(176,224,230,0.2)"
        },
        "stormy": {
            "--bg-color": "#0d1b2a", "--text-color": "#ff6b6b", 
            "--accent-color": "#e63946", "--card-bg": "rgba(230,57,70,0.15)"
        },
        "hot-day": {
            "--bg-color": "#ffd166", "--text-color": "#3d348b", 
            "--accent-color": "#f18701", "--card-bg": "rgba(241,135,1,0.2)"
        },
        "sunny-day": {
            "--bg-color": "#f9dbbd", "--text-color": "#6a4c93", 
            "--accent-color": "#ffa62b", "--card-bg": "rgba(255,166,43,0.2)"
        },
    }
    return themes.get(context, themes["default"])

def get_featured_project(context: str) -> dict:
    """Returns a full project object based on context."""
    project_map = {
        "night": 0,  # Neural Style Transfer
        "rainy": 1,  # Predictive Maintenance
        "stormy": 2,  # Sentiment Analysis
        "hot-day": 0,
        "sunny-day": 1,
        "default": 2,
    }
    project_index = project_map.get(context, 0)
    return projects_data[project_index]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)