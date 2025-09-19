# netlify-functions/api.py
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
from datetime import datetime
import csv

# Add the root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your FastAPI app from the main file
from main import app  # Assuming your main file is named main.py

# A simple handler for Netlify's serverless function
def handler(event, context):
    uvicorn.run(app, host="0.0.0.0", port=8000)
    return {
        'statusCode': 200,
        'body': 'Function is running'
    }