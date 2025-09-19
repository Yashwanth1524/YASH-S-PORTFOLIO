# netlify-functions/api.py
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
from mangum import Mangum   # 👈 NEW for Netlify Functions

# Initialize FastAPI app
app = FastAPI(title="Living Portfolio API", version="1.0")

# (Keep all your routes, functions, and logic exactly the same...)

# 🔑 At the very bottom, instead of uvicorn.run:
handler = Mangum(app)
