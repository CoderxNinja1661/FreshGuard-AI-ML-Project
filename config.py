"""
FreshGuard AI - Configuration Module
Intelligent Food Freshness Detection and Classification System
Student: Hariom Gupta (UID: 25MCA20022)
"""

import os
from dotenv import load_dotenv

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file
load_dotenv(os.path.join(BASE_DIR, '.env'))

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'freshguard_super_secret_mca_key_2026_hariom')
    
    # MySQL Database Settings (Strictly MySQL only - SQLite prohibited)
    DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
    DB_PORT = int(os.getenv('DB_PORT', '3307'))
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'freshguard_db')
    
    # Upload Settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    
    # Dataset and Model Paths
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'svm_model.pkl')
    KNN_MODEL_PATH = os.path.join(MODELS_DIR, 'knn_model.pkl')
    METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    
    # Freshness Classification Threshold
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.60'))
    
    # Student and Project Details
    PROJECT_NAME = "FreshGuard AI"
    STUDENT_NAME = "Hariom Gupta"
    STUDENT_UID = "25MCA20022"
    PROJECT_VERSION = "1.0.0"
