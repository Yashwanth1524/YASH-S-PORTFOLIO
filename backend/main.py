import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime
import pytz
from transformers import pipeline
import torch
from pydantic import BaseModel
from typing import Union
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -----------------------
# Configuration for email
# -----------------------
EMAIL_ADDRESS = "yashwanth150204@gmail.com"
EMAIL_PASSWORD = "kosd upbz ptym wqgy"  # Use Gmail App Password

# -----------------------
# Initialize FastAPI app
# -----------------------
app = FastAPI(title="Living Portfolio API", version="1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI pipelines
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    text_generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Error loading models: {e}")
    sentiment_analyzer = None
    text_generator = None

# Mount static files
os.makedirs("static/cleaned_images", exist_ok=True)
os.makedirs("static/uploaded_images", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------
# Redirect to HuggingFace space
# -----------------------
@app.get("/denoise-demo/")
async def redirect_to_hf_space():
    return RedirectResponse("https://huggingface.co/spaces/racecourse/Denoising")

# -----------------------
# Root
# -----------------------
@app.get("/")
async def root():
    return {"message": "Living Portfolio API is running!"}

# -----------------------
# Projects
# -----------------------
class Project(BaseModel):
    id: int
    title: str
    description: str
    tags: list
    github_url: str
    live_url: str = None

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
        "live_url": "https://huggingface.co/spaces/racecourse/Denoising"
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

# -----------------------
# Contact form (send email)
# -----------------------
class ContactForm(BaseModel):
    name: str
    email: str
    subject: str
    body: str

@app.post("/send-email/")
async def send_email(contact_form: ContactForm):
    try:
        # Compose the email
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_ADDRESS  # sending to yourself
        msg['Subject'] = f"Portfolio Inquiry: {contact_form.subject}"

        body = f"""
        Name: {contact_form.name}
        Email: {contact_form.email}
        Message: {contact_form.body}
        """
        msg.attach(MIMEText(body, 'plain'))

        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        return {"message": "Email sent successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

# -----------------------
# Sentiment Analysis
# -----------------------
@app.post("/analyze-sentiment/")
async def analyze_sentiment(text: str):
    if not sentiment_analyzer:
        return {"sentiment": "neutral", "confidence": 0.0}
    try:
        result = sentiment_analyzer(text[:512])[0]
        return {"sentiment": result['label'].lower(), "confidence": float(result['score'])}
    except Exception as e:
        return {"sentiment": "neutral", "confidence": 0.0}

# -----------------------
# Location & Context
# -----------------------
class LocationData(BaseModel):
    latitude: float
    longitude: float

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
        "default": {"--bg-color": "#1a1a2e", "--text-color": "#e6e6e6", "--accent-color": "#4cc9f0", "--card-bg": "rgba(255,255,255,0.1)"},
        "night": {"--bg-color": "#0f0f1f", "--text-color": "#a0a0d0", "--accent-color": "#7b68ee", "--card-bg": "rgba(160,160,208,0.15)"},
        "rainy": {"--bg-color": "#2b4162", "--text-color": "#f0f8ff", "--accent-color": "#a0d2db", "--card-bg": "rgba(176,224,230,0.2)"},
        "stormy": {"--bg-color": "#0d1b2a", "--text-color": "#ff6b6b", "--accent-color": "#e63946", "--card-bg": "rgba(230,57,70,0.15)"},
        "hot-day": {"--bg-color": "#ffd166", "--text-color": "#3d348b", "--accent-color": "#f18701", "--card-bg": "rgba(241,135,1,0.2)"},
        "sunny-day": {"--bg-color": "#f9dbbd", "--text-color": "#6a4c93", "--accent-color": "#ffa62b", "--card-bg": "rgba(255,166,43,0.2)"},
    }
    return themes.get(context, themes["default"])

def get_featured_project(context: str) -> dict:
    project_map = {"night": 0, "rainy": 1, "stormy": 2, "hot-day": 0, "sunny-day": 1, "default": 2}
    project_index = project_map.get(context, 0)
    return projects_data[project_index]

@app.post("/get-context/")
async def get_context(location: LocationData):
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={location.latitude}&longitude={location.longitude}&current_weather=true"
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()
        current_weather = weather_data['current_weather']

        is_day = current_weather['is_day']
        temperature = current_weather['temperature']
        weather_code = current_weather['weathercode']

        context = "default"
        if is_day == 0:
            context = "night"
        elif 51 <= weather_code < 80:
            context = "rainy"
        elif weather_code >= 95:
            context = "stormy"
        elif temperature > 30:
            context = "hot-day"
        else:
            context = "sunny-day"

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

# -----------------------
# Run Uvicorn
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
